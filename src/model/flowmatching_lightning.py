import torch
import time
import torch.nn as nn
import pytorch_lightning as pl

from src.model.AdvancedUNet2D import AdvancedUNet2D
from src.utils import topoloss_normal
from src.utils.io_handler import save_masks_to_disk, load_masks_from_disk, save_evaluation_metrics
from src.utils.evaluation import evaluate


class FlowMatchingModelLightning(pl.LightningModule):
    def __init__(self, in_channels: int, image_dim: int, pos_encoding_dim: int, num_class_embeds: int, learning_rate: float = 1e-4,
                 use_topo_constraint=False, use_topo_loss=False, topo_dim : int = 0, reg_weight : float = 1e-5):
        """
        A pytorch lightning module for training a flow matching model.
        :param in_channels: Number of channels in the input image
        :param image_dim: Resolution of the input images
        :param pos_encoding_dim:  Dimension of the positional encoding
        :param num_class_embeds: Number of class embeddings (unused).
        :param learning_rate: Learning rate for the Adam optimizer
        :param use_topo_constraint: Whether to use topological constraint or not
        :param use_topo_loss: Whether to use topological loss or not
        :param topo_dim: Betti number to condition the model on (e.g. 0 for connected components, or 1 for distinct areas)
        :param reg_weight: Regularization weight used for the topological loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.image_dim = image_dim
        self.learning_rate = learning_rate
        self.use_topo_constraint = use_topo_constraint
        self.use_topo_loss = use_topo_loss
        self.reg_weight = reg_weight

        self.pos_encoding_dim = pos_encoding_dim
        self.num_class_embeds = num_class_embeds

        block_out_channels = (64, 128, 128, 256) if image_dim == 64 else (128, 128, 256, 256, 512, 512)
        down_block_type = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D") if image_dim == 64 else\
            ("DownBlock2D","DownBlock2D","DownBlock2D",  "DownBlock2D","AttnDownBlock2D","AttnDownBlock2D")
        up_block_type = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D") if image_dim == 64 else\
            ("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D")


        self.unet = AdvancedUNet2D(
            use_topo_embedding=True,
            sample_size=self.image_dim,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_type,
            up_block_types=up_block_type,
        )

        self.loss_function = nn.MSELoss()
        self.topo_loss = topoloss_normal.TopoLossMSE2D(True, True, True, topo_dim)
        self.topo_dim = topo_dim

        print(f"[INFO] Using topological constraint: {use_topo_constraint}")
        print(f"[INFO] Using topological loss: {use_topo_loss}")


    def training_step(self, batch, batch_idx):
        """
        Perform one training step
        :param batch: Training batch containing especially: Images and Betti Numbers
        :param batch_idx: Batch index
        """

        start = time.time()
        betti_numbers = None
        if len(batch) == 4:
            images, labels, betti_numbers, _ = batch
        else:
            images, labels = batch

        noise = torch.randn_like(images)

        ts = torch.rand(size=(images.shape[0],), device=self.device)
        ts_view = ts.view(-1, 1, 1, 1)

        x = (1-ts_view)  * noise + ts_view * images
        real_diff = images - noise

        betti_numbers = betti_numbers if self.use_topo_constraint or self.use_topo_loss else None
        flow_pred = self.unet(x, ts, topo_constraints=betti_numbers).sample

        loss = self.loss_function(real_diff, flow_pred)

        x0_pred = (noise + flow_pred)

        if self.use_topo_loss:
            topo_loss = self.topo_loss(x0_pred.clamp(-1, 1), images)
            loss = loss + self.reg_weight * topo_loss.mean()
            self.log("topo_loss", topo_loss.mean(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss


    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int, labels=None, betti_numbers=None):
        """
        Samples images from pure noise.
        :param num_samples: number of images to sample
        :param num_steps: number steps used during sampling.
        :param labels: Unused. Kept for compatability.
        :param betti_numbers: The Betti number constraint for reach image.
        :return:
        """
        x = torch.randn((num_samples, self.in_channels, self.image_dim, self.image_dim), device=self.device) # pure noise

        del_t = 1 / num_steps
        for i in range(num_steps):
            ts = torch.full((x.shape[0],), del_t * i, device=self.device, dtype=x.dtype)

            betti_numbers = betti_numbers if self.use_topo_constraint or self.use_topo_loss else None
            flow = self.unet(x, ts, topo_constraints=betti_numbers).sample
            x = x + del_t * flow

        return x.clamp(-1, 1)

    # just for compatability
    @torch.no_grad()
    def sample_ddim(self, num_samples, num_steps=0, eta=0.0, labels=None, betti_numbers=None):
        return self.sample(num_samples, num_steps, labels, betti_numbers)


    def sample_improve(self, x0, betti_numbers, num_steps=10):
        """
            Sampling method that keeps gradient.
        """
        self.eval()

        del_t = 1 / num_steps
        x = x0
        for i in range(num_steps):
            x = self._sample_improve_step(x, betti_numbers, i, del_t)

        return x.clamp(-1, 1)

    def _sample_improve_step(self, x, betti_numbers, step, del_t):
        """
         Performs on sampling step and keeps gradient.
        """
        ts = torch.full((x.shape[0],), del_t * step, device=self.device, dtype=x.dtype)

        betti_numbers = betti_numbers if self.use_topo_constraint or self.use_topo_loss else None
        flow = self.unet(x, ts, topo_constraints=betti_numbers).sample
        x = x + del_t * flow

        return x


    def _predict_x1_from_xt(self, x_t, betti_numbers, step, num_total_steps):
        """
        Predict x1 (clean image) directly from xt (noisy image) in one step.
        :param x_t: noisy image
        :param betti_numbers: The Betti number constraint for reach image.
        :param step: The current timestep of the noisy images
        :param num_total_steps: Total number of timesteps used for sampling.
        :return: predict clean images
        """
        # timestep t is given as int
        t = float(step) / float(num_total_steps)

        ts = torch.full((x_t.shape[0],), t , device=self.device, dtype=x_t.dtype)
        betti_numbers = betti_numbers if self.use_topo_constraint or self.use_topo_loss else None
        flow = self.unet(x_t, ts, topo_constraints=betti_numbers).sample

        x_1 = x_t + (1.0-t) * flow

        return x_1.clamp(-1, 1)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)