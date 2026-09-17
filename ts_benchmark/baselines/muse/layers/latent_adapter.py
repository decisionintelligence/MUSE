"""Channel-token latent adapter for MUSE patch tokens."""

import torch
from torch import nn
from torch.nn import functional as F


class VariableAwareLatentAdapter(nn.Module):
    """Use channel tokens to exchange information between variables."""

    def __init__(
        self,
        embed_dim=768,
        num_latents=1,
        latent_dim=192,
        num_heads=4,
        channel_depth=1,
    ):
        super().__init__()

        if num_latents != 1:
            raise ValueError("Channel-token adapter requires num_latents=1.")
        if not isinstance(channel_depth, int) or isinstance(channel_depth, bool) \
                or channel_depth < 1:
            raise ValueError("channel_depth must be a positive integer.")
        self.num_latents = num_latents
        self.channel_depth = channel_depth
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.patch_down = nn.Linear(embed_dim, latent_dim)

        self.latent_queries = nn.Parameter(
            torch.zeros(1, num_latents, latent_dim)
        )
        self.latent_query_norm = nn.LayerNorm(latent_dim)
        self.patch_memory_norm = nn.LayerNorm(latent_dim)
        self.latent_cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )
        self.channel_query_norm = nn.LayerNorm(latent_dim)
        self.channel_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )
        self.channel_memory_norm = nn.LayerNorm(latent_dim)

        self.patch_query_norm = nn.LayerNorm(latent_dim)
        self.latent_memory_norm = nn.LayerNorm(latent_dim)
        self.patch_cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )

        self.patch_up = nn.Linear(latent_dim, embed_dim)

        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        nn.init.zeros_(self.patch_up.weight)
        nn.init.zeros_(self.patch_up.bias)

    def project_patch_memory(self, patch_tokens):
        patches = self.patch_down(self.patch_norm(patch_tokens))
        return self.patch_memory_norm(patches)

    def initial_channel_tokens(self, batch_size, num_variables):
        return self.latent_queries.expand(batch_size, num_variables, -1)

    def update_variable_tokens(self, patch_memory, channel_tokens):
        batch_size, num_variables, num_patches, latent_dim = (
            patch_memory.shape
        )
        sequence = torch.cat((
            channel_tokens.unsqueeze(2), patch_memory
        ), dim=2).reshape(
            batch_size * num_variables, num_patches + 1, latent_dim
        )
        normalized = self.latent_query_norm(sequence)
        update = self.latent_cross_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        sequence = self.latent_memory_norm(sequence + update).reshape(
            batch_size, num_variables, num_patches + 1, latent_dim
        )
        return sequence[:, :, 1:, :], sequence[:, :, 0, :]

    def mix_channel_tokens(self, channel_tokens):
        normalized = self.channel_query_norm(channel_tokens)
        update = self.channel_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return self.channel_memory_norm(channel_tokens + update)

    def shared_correction(self, latent_memory):
        attention = self.patch_cross_attention
        if attention.training and attention.dropout != 0:
            raise RuntimeError(
                "Single-source attention shortcut requires zero dropout."
            )
        if attention.bias_k is not None or attention.bias_v is not None:
            raise RuntimeError(
                "Single-source attention shortcut does not support bias_kv."
            )
        if attention.add_zero_attn:
            raise RuntimeError(
                "Single-source attention shortcut does not support zero attention."
            )

        embed_dim = attention.embed_dim
        if attention._qkv_same_embed_dim:
            value_weight = attention.in_proj_weight[2 * embed_dim:]
        else:
            value_weight = attention.v_proj_weight
        value_bias = (
            None if attention.in_proj_bias is None
            else attention.in_proj_bias[2 * embed_dim:]
        )
        correction = F.linear(latent_memory, value_weight, value_bias)
        correction = attention.out_proj(correction)
        correction = self.patch_up(correction)
        return correction.unsqueeze(2)

    def correction_from_patch_memory(self, patch_memory):
        batch_size, num_variables = patch_memory.shape[:2]
        channel_tokens = self.initial_channel_tokens(
            batch_size, num_variables
        )
        for _ in range(self.channel_depth):
            patch_memory, channel_tokens = self.update_variable_tokens(
                patch_memory, channel_tokens
            )
            channel_tokens = self.mix_channel_tokens(channel_tokens)
        return self.shared_correction(channel_tokens)

    @staticmethod
    def apply_correction(patch_tokens, correction):
        return patch_tokens + correction

    def forward(self, patch_tokens, return_correction=False):
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        patch_memory = self.project_patch_memory(patch_tokens)
        correction = self.correction_from_patch_memory(patch_memory)
        output = self.apply_correction(patch_tokens, correction)
        if return_correction:
            return output, correction
        return output


class RelativePositionAttention(nn.Module):
    def __init__(self, dim, num_heads, num_offsets):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Attention dimension must be divisible by heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, num_offsets))

    def forward(self, x, bias_index):
        leading_shape = x.shape[:-2]
        length, dim = x.shape[-2:]
        x = x.reshape(-1, length, dim)
        q = self.q_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).reshape(
            x.shape[0], length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + self.relative_bias[:, bias_index].unsqueeze(0)
        weights = logits.softmax(dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(
            x.shape[0], length, dim
        )
        output = self.out_proj(output).reshape(*leading_shape, length, dim)
        return output


class TemporalPeriodicAdapter(nn.Module):
    def __init__(
        self, embed_dim=768, bottleneck_dim=64, num_heads=4,
        num_rows=14, num_columns=1
    ):
        super().__init__()
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.center_norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.temporal_attention = RelativePositionAttention(
            bottleneck_dim, num_heads, 2 * num_columns - 1
        )
        self.periodic_attention = RelativePositionAttention(
            bottleneck_dim, num_heads, num_rows
        )
        self.output_norm = nn.LayerNorm(bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        columns = torch.arange(num_columns)
        rows = torch.arange(num_rows)
        self.register_buffer(
            "temporal_bias_index",
            columns[:, None] - columns[None, :] + num_columns - 1,
        )
        self.register_buffer(
            "periodic_bias_index",
            (rows[:, None] - rows[None, :]) % num_rows,
        )

    def forward(self, patch_tokens, residual_gate):
        batch_size, num_variables, num_patches, embed_dim = patch_tokens.shape
        if num_patches != self.num_rows * self.num_columns:
            raise ValueError("Unexpected visible patch geometry.")
        grid = patch_tokens.reshape(
            batch_size, num_variables, self.num_rows,
            self.num_columns, embed_dim
        )
        centered = grid - grid.mean(dim=(2, 3), keepdim=True)
        hidden = self.down(self.center_norm(centered))
        temporal = hidden.reshape(
            batch_size, num_variables, self.num_rows,
            self.num_columns, hidden.shape[-1]
        )
        temporal_update = self.temporal_attention(
            temporal, self.temporal_bias_index
        )
        temporal = temporal + temporal_update
        periodic = temporal.permute(0, 1, 3, 2, 4)
        periodic_update = self.periodic_attention(
            periodic, self.periodic_bias_index
        )
        periodic = periodic + periodic_update
        hidden = periodic.permute(0, 1, 3, 2, 4)
        raw_correction = self.up(self.output_norm(hidden))
        applied_correction = residual_gate * raw_correction
        output = grid + applied_correction
        return output.reshape_as(patch_tokens)
