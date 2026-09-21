""" A simple ema callback for pytorch lightning """

import pytorch_lightning as pl

class EMACallback(pl.Callback):

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_params: dict = {}
        self._backup_params: dict = {}


    def on_train_start(self, trainer, pl_module):
        if self.ema_params:
            device = next(pl_module.parameters()).device
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.ema_params:
                    self.ema_params[name] = param.data.clone()
                else:
                    self.ema_params[name] = self.ema_params[name].to(device)
            return

        self.ema_params = {
            name: param.data.clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            self.ema_params[name].mul_(self.decay).add_(
                param.data, alpha=1.0 - self.decay
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._swap_to_train(pl_module)


    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_params"] = {
            k: v.cpu() for k, v in self.ema_params.items()
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_params" in checkpoint:
            device = next(pl_module.parameters()).device
            self.ema_params = {
                k: v.to(device) for k, v in checkpoint["ema_params"].items()
            }


    def use_ema(self, pl_module):
        return _EMAContext(self, pl_module)


    def _swap_to_ema(self, pl_module):
        self._backup_params = {}
        for name, param in pl_module.named_parameters():
            if name in self.ema_params:
                self._backup_params[name] = param.data.clone()
                param.data.copy_(self.ema_params[name])

    def _swap_to_train(self, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self._backup_params:
                param.data.copy_(self._backup_params[name])
        self._backup_params = {}


class _EMAContext:

    def __init__(self, callback: EMACallback, pl_module):
        self.cb = callback
        self.module = pl_module

    def __enter__(self):
        self.cb._swap_to_ema(self.module)
        return self.module

    def __exit__(self, *args):
        self.cb._swap_to_train(self.module)
