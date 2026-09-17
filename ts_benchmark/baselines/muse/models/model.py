import torch

from pathlib import Path

from . import models_mae
from ..layers.latent_adapter import (
    TemporalPeriodicAdapter,
    VariableAwareLatentAdapter,
)
import einops
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from PIL import Image
from ..utils import util


MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}


class MUSE(nn.Module):
    variable_chunk_size = 16

    def __init__(self, arch='mae_base', ckpt_path=None, load_ckpt=True,
                 num_latents=1, latent_dim=192, adapter_num_heads=4,
                 channel_depth=1):
        super(MUSE, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")
        if not isinstance(channel_depth, int) or isinstance(channel_depth, bool) \
                or channel_depth < 1:
            raise ValueError("channel_depth must be a positive integer.")

        self.vision_model = MAE_ARCH[arch][0]()

        if load_ckpt:
            if ckpt_path is None:
                ckpt_path = (
                    Path(__file__).resolve().parents[4]
                    / "pretrained_weights"
                    / "mae"
                    / MAE_ARCH[arch][1]
                )
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.vision_model.load_state_dict(checkpoint['model'], strict=True)

        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.variable_adapter = VariableAwareLatentAdapter(
            embed_dim=self.vision_model.pos_embed.shape[-1],
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_heads=adapter_num_heads,
            channel_depth=channel_depth,
        )
        self.fusion_logit = nn.Parameter(torch.zeros(()))
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const

        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()
        if self.num_patch != 14 or len(self.vision_model.blocks) != 12:
            raise ValueError("Temporal-periodic branch requires mae_base geometry.")
        self.temporal_periodic_adapters = nn.ModuleList([
            TemporalPeriodicAdapter(
                embed_dim=self.vision_model.pos_embed.shape[-1],
                bottleneck_dim=64,
                num_heads=4,
                num_rows=14,
                num_columns=self.num_patch_input,
            )
            for _ in range(3)
        ])
        self.tp_residual_logits = nn.Parameter(torch.full((12,), -2.944))

    def _normalize_series(self, x, fp64=False):
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(
                x_enc.to(torch.float64) if fp64 else x_enc,
                dim=1, keepdim=True, unbiased=False
            ) + 1e-5
        )
        stdev /= self.norm_const
        x_enc /= stdev
        return einops.rearrange(x_enc, 'b s n -> b n s'), means, stdev

    def _render_variable_images(self, x_enc):
        batch_size, num_variables, _ = x_enc.shape
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')
        x_2d = einops.rearrange(
            x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity
        )
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros(
            (
                batch_size * num_variables, 1, self.image_size,
                self.num_patch_output * self.patch_size
            ),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        image_input = torch.cat((x_resize, masked), dim=-1)
        return einops.repeat(
            image_input, 'b 1 h w -> b c h w', c=3
        )

    def _shared_mask_noise(self):
        ids_shuffle = torch.argsort(self.mask, dim=1)
        ranks = torch.arange(
            self.mask.shape[1], device=self.mask.device, dtype=self.mask.dtype
        ).unsqueeze(0)
        return torch.empty_like(self.mask).scatter_(1, ids_shuffle, ranks)

    def _visible_grid_permutations(self, noise):
        visible_positions = torch.argsort(noise, dim=1)[
            :, :self.num_patch * self.num_patch_input
        ]
        to_grid = torch.argsort(visible_positions, dim=1)[0]
        return to_grid, torch.argsort(to_grid)

    def _visible_patch_chunk(self, x_enc, noise):
        batch_size, num_variables, _ = x_enc.shape
        image_input = self._render_variable_images(x_enc)
        visible, _, _ = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            noise.expand(batch_size * num_variables, -1),
        )
        return visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )

    def _channel_token_chunk(self, x_enc, noise, *channel_history):
        visible = self._visible_patch_chunk(x_enc, noise)
        patch_memory = self.variable_adapter.project_patch_memory(visible)
        for channel_tokens in channel_history:
            patch_memory, channel_tokens = (
                self.variable_adapter.update_variable_tokens(
                    patch_memory, channel_tokens
                )
            )
        return channel_tokens

    def prepare_variable_chunk_context(self, x, fp64=False):
        x_enc, means, stdev = self._normalize_series(x, fp64=fp64)
        num_variables = x_enc.shape[1]
        noise = self._shared_mask_noise()
        channel_history = [
            self.variable_adapter.initial_channel_tokens(
                x_enc.shape[0], num_variables
            )
        ]
        for _ in range(self.variable_adapter.channel_depth):
            token_chunks = []
            for start in range(0, num_variables, self.variable_chunk_size):
                end = min(start + self.variable_chunk_size, num_variables)
                variable_chunk = x_enc[:, start:end, :]
                history_chunk = [
                    channel_tokens[:, start:end, :]
                    for channel_tokens in channel_history
                ]
                if self.training and torch.is_grad_enabled():
                    token_chunk = checkpoint(
                        self._channel_token_chunk,
                        variable_chunk,
                        noise,
                        *history_chunk,
                        use_reentrant=False,
                    )
                else:
                    token_chunk = self._channel_token_chunk(
                        variable_chunk, noise, *history_chunk
                    )
                token_chunks.append(token_chunk)
            channel_local = torch.cat(token_chunks, dim=1)
            channel_tokens = self.variable_adapter.mix_channel_tokens(
                channel_local
            )
            channel_history.append(channel_tokens)
        correction = self.variable_adapter.shared_correction(channel_tokens)
        return {
            "x_enc": x_enc,
            "means": means,
            "stdev": stdev,
            "noise": noise,
            "grid_permutations": self._visible_grid_permutations(noise),
            "correction": correction,
            "num_variables": num_variables,
        }

    def _sequence_with_cls(self, patch_tokens):
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        patches = patch_tokens.reshape(
            batch_size * num_variables, num_patches, embed_dim
        )
        cls_token = self.vision_model.cls_token + self.vision_model.pos_embed[:, :1]
        return torch.cat((cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)

    def _encode_branches(self, visible, grid_permutations, correction=None):
        if correction is None:
            channel_patches, correction = self.variable_adapter(
                visible, return_correction=True
            )
        else:
            channel_patches = self.variable_adapter.apply_correction(
                visible, correction
            )
        channel_latent = self._sequence_with_cls(channel_patches)
        for block in self.vision_model.blocks:
            channel_latent = block(channel_latent)
        channel_latent = self.vision_model.norm(channel_latent)

        tp_latent = self._sequence_with_cls(visible)
        to_grid, from_grid = grid_permutations
        for layer, block in enumerate(self.vision_model.blocks):
            tp_latent = block(tp_latent)
            patches = tp_latent[:, 1:].reshape_as(visible)
            grid_patches = patches.index_select(2, to_grid)
            grid_patches = self.temporal_periodic_adapters[layer // 4](
                grid_patches,
                torch.sigmoid(self.tp_residual_logits[layer]),
            )
            patches = grid_patches.index_select(2, from_grid)
            tp_latent = torch.cat((
                tp_latent[:, :1],
                patches.reshape(tp_latent.shape[0], patches.shape[2], patches.shape[3]),
            ), dim=1)
        tp_latent = self.vision_model.norm(tp_latent)
        return channel_latent, tp_latent, correction

    def _decode_normalized(
        self, latent, ids_restore, batch_size, num_variables,
        return_image=False
    ):
        prediction = self.vision_model.forward_decoder(latent, ids_restore)
        image = self.vision_model.unpatchify(prediction)
        segments = self.output_resize(image.mean(1, keepdim=True))
        flattened = einops.rearrange(
            segments, '(b n) 1 f p -> b (p f) n',
            b=batch_size, f=self.periodicity
        )
        normalized = flattened[
            :,
            self.pad_left + self.context_len:
            self.pad_left + self.context_len + self.pred_len,
            :,
        ]
        if return_image:
            return normalized, image
        return normalized

    def _forward_visible(
        self, visible, ids_restore, grid_permutations, means, stdev,
        correction=None,
        return_image=False, return_branches=False
    ):
        batch_size, num_variables = visible.shape[:2]
        channel_latent, tp_latent, correction = self._encode_branches(
            visible, grid_permutations, correction=correction
        )
        channel_decoded = self._decode_normalized(
            channel_latent, ids_restore, batch_size, num_variables,
            return_image=return_image
        )
        tp_decoded = self._decode_normalized(
            tp_latent, ids_restore, batch_size, num_variables,
            return_image=return_image
        )
        if return_image:
            channel_normalized, channel_image = channel_decoded
            tp_normalized, tp_image = tp_decoded
        else:
            channel_normalized = channel_decoded
            tp_normalized = tp_decoded
        gate = torch.sigmoid(self.fusion_logit)
        fused_normalized = (
            gate * channel_normalized + (1 - gate) * tp_normalized
        )
        result = {
            "output": fused_normalized * stdev + means,
        }
        if return_branches:
            result.update({
                "channel": channel_normalized * stdev + means,
                "tp": tp_normalized * stdev + means,
            })
        if return_image:
            result["image"] = gate * channel_image + (1 - gate) * tp_image
        return result

    def forward_variable_chunk(
        self, context, start, end, correction=None, export_image=False,
        return_branches=False
    ):
        x_enc = context["x_enc"][:, start:end, :]
        batch_size, num_variables, _ = x_enc.shape
        image_input = self._render_variable_images(x_enc)
        visible, mask, ids_restore = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            context["noise"].expand(batch_size * num_variables, -1),
        )
        visible = visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )
        if correction is None:
            correction = context["correction"]
        correction = correction[:, start:end]
        result = self._forward_visible(
            visible, ids_restore, context["grid_permutations"],
            context["means"][:, :, start:end],
            context["stdev"][:, :, start:end],
            correction=correction,
            return_image=export_image,
            return_branches=return_branches,
        )
        if export_image:
            mask_image = self.vision_model.unpatchify(
                mask.detach().unsqueeze(-1).repeat(
                    1, 1, self.patch_size ** 2 * 3
                )
            )
            reconstructed = (
                image_input * (1 - mask_image) + result["image"] * mask_image
            )
            green_bg = -torch.ones_like(image_input) * 2
            masked_input = image_input * (1 - mask_image) + green_bg * mask_image
            result["input_image"] = einops.rearrange(
                masked_input, '(b n) c h w -> b n h w c', b=batch_size
            )
            result["reconstructed_image"] = einops.rearrange(
                reconstructed, '(b n) c h w -> b n h w c', b=batch_size
            )
        if return_branches:
            return result
        if export_image:
            return (
                result["output"], result["input_image"],
                result["reconstructed_image"]
            )
        return result["output"]

    def _forward_variable_chunks(
        self, x, export_image=False, fp64=False, return_branches=False
    ):
        context = self.prepare_variable_chunk_context(x, fp64=fp64)
        results = []
        for start in range(
            0, context["num_variables"], self.variable_chunk_size
        ):
            end = min(
                start + self.variable_chunk_size,
                context["num_variables"],
            )
            results.append(self.forward_variable_chunk(
                context, start, end, export_image=export_image,
                return_branches=return_branches,
            ))
        if not return_branches:
            if export_image:
                return (
                    torch.cat([item[0] for item in results], dim=-1),
                    torch.cat([item[1] for item in results], dim=1),
                    torch.cat([item[2] for item in results], dim=1),
                )
            return torch.cat(results, dim=-1)
        merged = {
            name: torch.cat([item[name] for item in results], dim=-1)
            for name in ("output", "channel", "tp")
        }
        if export_image:
            merged["input_image"] = torch.cat(
                [item["input_image"] for item in results], dim=1
            )
            merged["reconstructed_image"] = torch.cat(
                [item["reconstructed_image"] for item in results], dim=1
            )
        return merged

    def forward(
        self, x, export_image=False, fp64=False, use_variable_chunk=False,
        return_branches=False
    ):
        if use_variable_chunk:
            return self._forward_variable_chunks(
                x, export_image=export_image, fp64=fp64,
                return_branches=return_branches,
            )
        x_enc, means, stdev = self._normalize_series(x, fp64=fp64)
        batch_size, num_variables = x_enc.shape[:2]
        image_input = self._render_variable_images(x_enc)
        noise = self._shared_mask_noise()
        visible, mask, ids_restore = self.vision_model.prepare_visible_patches(
            image_input,
            self.mask_ratio,
            noise.expand(batch_size * num_variables, -1),
        )
        visible = visible.reshape(
            batch_size, num_variables, visible.shape[1], visible.shape[2]
        )
        result = self._forward_visible(
            visible, ids_restore, self._visible_grid_permutations(noise),
            means, stdev,
            return_image=export_image,
            return_branches=return_branches,
        )
        if export_image:
            mask_image = self.vision_model.unpatchify(
                mask.detach().unsqueeze(-1).repeat(
                    1, 1, self.patch_size ** 2 * 3
                )
            )
            reconstructed = (
                image_input * (1 - mask_image) + result["image"] * mask_image
            )
            green_bg = -torch.ones_like(image_input) * 2
            masked_input = image_input * (1 - mask_image) + green_bg * mask_image
            result["input_image"] = einops.rearrange(
                masked_input, '(b n) c h w -> b n h w c', b=batch_size
            )
            result["reconstructed_image"] = einops.rearrange(
                reconstructed, '(b n) c h w -> b n h w c', b=batch_size
            )
        if return_branches:
            return result
        if export_image:
            output = (
                result["output"], result["input_image"],
                result["reconstructed_image"]
            )
        else:
            output = result["output"]
        return output
