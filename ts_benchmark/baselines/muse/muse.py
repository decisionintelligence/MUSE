import pandas as pd
import torch
from torch.utils.data import DataLoader

from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
)
from ts_benchmark.utils.get_device import get_device

from .models.model import MUSE as MUSEModel
from .utils.util import freq_to_seasonality_list


MODEL_HYPER_PARAMS = {
    "arch": "mae_base",
    "ckpt_path": None,
    "load_ckpt": True,
    "periodicity": "auto",
    "norm_const": 0.4,
    "align_const": 0.4,
    "interpolation": "bilinear",
    "num_latents": 1,
    "latent_dim": 192,
    "adapter_num_heads": 4,
    "channel_depth": 1,
    "use_variable_chunk": False,
    "fp64": False,
    "batch_size": 32,
    "num_epochs": 1,
}


class MUSE(DeepForecastingModelBase):
    """TFB adapter for globally corrected MUSE patch tokens."""

    def __init__(self, **kwargs):
        super(MUSE, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self._series_dim = None

    @property
    def model_name(self):
        return "MUSE"

    def _init_criterion_and_optimizer(self):
        gate = self._core_model().fusion_logit
        adapter_parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad and parameter is not gate
        ]
        optimizer = torch.optim.Adam([
            {
                "params": adapter_parameters,
                "lr": self.config.lr,
                "parameter_group": "adapters",
            },
            {
                "params": [gate],
                "lr": 10 * self.config.lr,
                "weight_decay": 0.0,
                "parameter_group": "gate",
            },
        ])
        return torch.nn.MSELoss(), optimizer

    def _adjust_lr(self, optimizer, epoch, config):
        super()._adjust_lr(optimizer, epoch, config)
        adapter_lr = next(
            group["lr"] for group in optimizer.param_groups
            if group["parameter_group"] == "adapters"
        )
        for group in optimizer.param_groups:
            if group["parameter_group"] == "gate":
                group["lr"] = 10 * adapter_lr

    def _init_model(self):
        checkpoint_path = getattr(self.config, "checkpoint_path", None)
        if checkpoint_path is None:
            checkpoint_path = self.config.ckpt_path
        model = MUSEModel(
            arch=self.config.arch,
            ckpt_path=checkpoint_path,
            load_ckpt=self.config.load_ckpt,
            num_latents=self.config.num_latents,
            latent_dim=self.config.latent_dim,
            adapter_num_heads=self.config.adapter_num_heads,
            channel_depth=self.config.channel_depth,
        )
        model.update_config(
            context_len=self.config.seq_len,
            pred_len=self.config.horizon,
            periodicity=self.config.periodicity,
            norm_const=self.config.norm_const,
            align_const=self.config.align_const,
            interpolation=self.config.interpolation,
        )
        return model

    def _set_data_frequency(self, train_data):
        frequency = pd.infer_freq(train_data.index)
        if frequency is None:
            raise ValueError("Irregular time intervals")

        self.config.freq = frequency

        periodicity = self.config.periodicity
        if periodicity is None:
            periodicity = 0
        if isinstance(periodicity, str):
            value = periodicity.strip().lower()
            if value in {"auto", "freq"}:
                periodicity = 0
            else:
                try:
                    periodicity = int(value)
                except ValueError as error:
                    raise ValueError(
                        "periodicity must be a positive integer or 'auto'."
                    ) from error

        periodicity = int(periodicity)
        if periodicity == 0:
            periodicity = freq_to_seasonality_list(frequency)[0]
        elif periodicity < 0:
            raise ValueError("periodicity must be positive or 'auto'.")
        self.config.periodicity = periodicity

    def multi_forecasting_hyper_param_tune(self, train_data):
        super().multi_forecasting_hyper_param_tune(train_data)
        self._set_data_frequency(train_data)

    def single_forecasting_hyper_param_tune(self, train_data):
        super().single_forecasting_hyper_param_tune(train_data)
        self._set_data_frequency(train_data)

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv=1.0,
        **kwargs,
    ):
        self._series_dim = train_valid_data.shape[-1]
        if self.config.use_variable_chunk:
            result = self._forecast_fit_variable_chunks(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
            )
        else:
            result = super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                **kwargs,
            )
        return result

    def validate(self, valid_data_loader, series_dim, criterion):
        config = self.config
        valid_data_loader = DataLoader(
            valid_data_loader.dataset,
            batch_size=valid_data_loader.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        self.model.eval()
        device = get_device()
        fused_square_sum = 0.0
        value_count = 0
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target = input.to(device), target.to(device)
                output = self.model(
                    input,
                    fp64=config.fp64,
                    use_variable_chunk=config.use_variable_chunk,
                )
                target = target[:, -config.horizon:, :series_dim]
                output = output[:, -config.horizon:, :series_dim]
                output, target = self._post_process(output, target)
                fused_error = (output - target).double()
                fused_square_sum += fused_error.square().sum().item()
                value_count += fused_error.numel()
        val_mse_fused = fused_square_sum / value_count
        self.model.train()
        return val_mse_fused

    def _backward_variable_chunks(
        self, input, target, series_dim, criterion, scaler=None
    ):
        model = self._core_model()
        context = model.prepare_variable_chunk_context(
            input, fp64=self.config.fp64
        )
        correction = context["correction"]
        correction_leaf = correction.detach().requires_grad_(True)
        for start in range(
            0, context["num_variables"], model.variable_chunk_size
        ):
            end = min(
                start + model.variable_chunk_size,
                context["num_variables"],
            )
            supervised_end = min(end, series_dim)
            supervised_width = max(0, supervised_end - start)

            if supervised_width == 0:
                continue

            result = model.forward_variable_chunk(
                context,
                start,
                end,
                correction=correction_leaf,
                return_branches=True,
            )
            channel = result["channel"][
                :, -self.config.horizon:, :supervised_width
            ]
            tp = result["tp"][
                :, -self.config.horizon:, :supervised_width
            ]
            chunk_target = target[
                :, -self.config.horizon:, start:supervised_end
            ]
            channel, processed_target = self._post_process(
                channel, chunk_target
            )
            tp, _ = self._post_process(
                tp, chunk_target
            )
            gate = torch.sigmoid(model.fusion_logit)
            gate_prediction = (
                gate * channel.detach() + (1 - gate) * tp.detach()
            )
            weighted_loss = (
                criterion(channel, processed_target)
                + criterion(tp, processed_target)
                + criterion(gate_prediction, processed_target)
            ) * (
                supervised_width / series_dim
            )
            if scaler is None:
                weighted_loss.backward()
            else:
                scaler.scale(weighted_loss).backward()

        if correction_leaf.grad is not None and correction.requires_grad:
            correction.backward(correction_leaf.grad)

    def _forecast_fit_variable_chunks(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv=1.0,
    ):
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog")
        if exog_data is not None:
            train_valid_data = pd.concat(
                [train_valid_data, exog_data], axis=1
            )

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.model = self._init_model()
        print(
            "----------------------------------------------------------",
            self.model_name,
        )

        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )
        self.scaler.fit(train_data.values)
        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            _, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
            )

        _, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        criterion, optimizer = self._init_criterion_and_optimizer()
        grad_scaler = (
            torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        )
        device = get_device()
        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            for input, target, input_mark, target_mark in self.train_data_loader:
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                self._backward_variable_chunks(
                    input,
                    target,
                    series_dim,
                    criterion,
                    scaler=grad_scaler,
                )
                if grad_scaler is None:
                    optimizer.step()
                else:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                if config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(
                    valid_data_loader, series_dim, criterion
                )
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint(self.model)
                if self.early_stopping.early_stop:
                    break

            if config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

    def _core_model(self):
        return getattr(self.model, "module", self.model)

    def _process(self, input, target, input_mark, target_mark):
        separate_losses = self.model.training and torch.is_grad_enabled()
        result = self.model(
            input,
            fp64=self.config.fp64,
            use_variable_chunk=self.config.use_variable_chunk,
            return_branches=separate_losses,
        )
        if not separate_losses:
            return {"output": result}
        target = target[
            :, -self.config.horizon:, :self._series_dim
        ]
        channel, processed_target = self._post_process(
            result["channel"][:, :, :self._series_dim], target
        )
        tp, _ = self._post_process(
            result["tp"][:, :, :self._series_dim], target
        )
        gate = torch.sigmoid(self._core_model().fusion_logit)
        gate_prediction = gate * channel.detach() + (1 - gate) * tp.detach()
        tp_loss = torch.mean((tp - processed_target) ** 2)
        gate_loss = torch.mean((gate_prediction - processed_target) ** 2)
        return {
            "output": result["channel"],
            "additional_loss": tp_loss + gate_loss,
        }
