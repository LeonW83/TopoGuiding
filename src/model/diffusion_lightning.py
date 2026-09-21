import torch
import pytorch_lightning as pl

from src.model.AdvancedUNet2D import AdvancedUNet2D
from src.utils import topoloss_normal


class DiffusionModelLightning(pl.LightningModule):
    def __init__(self, in_channels: int, image_dim: int, pos_encoding_dim: int, num_class_embeds: int, timesteps: int, learning_rate: float = 1e-4,
                 use_topo_constraint: bool = True, use_topo_loss: bool = True, topo_dim: int = 0, reg_weight : float = 1e-5):
        """
        A pytorch lightning module for training a diffusion model.
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
        self.pos_encoding_dim = pos_encoding_dim
        self.timesteps = timesteps
        self.lr = learning_rate
        self.num_class_embeds=num_class_embeds
        self.use_topo_constraint = use_topo_constraint
        self.use_topo_loss = use_topo_loss
        self.reg_weight = reg_weight

        block_out_channels = (64, 128, 128, 256) if image_dim == 64 else (128, 128, 256, 256, 512, 512)
        down_block_type = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D") if image_dim == 64 else ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up_block_type = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D") if image_dim == 64 else ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")


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

        self.lr = learning_rate
        self.timesteps = timesteps
        self.topo_dim=topo_dim

        alphas, alphas_cumprod, betas = self.generate_alphas(timesteps)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer( "alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("betas", betas, persistent=False)

        self.loss_function = torch.nn.MSELoss()
        self.topo_loss = topoloss_normal.TopoLossMSE2D(True, True, True, topo_dim)

        print(f"[INFO] Using topological constraint: {use_topo_constraint}")
        print(f"[INFO] Using topological loss: {use_topo_loss}")


    def generate_alphas(self, timesteps):
        """
        Generate alphas using linear strategy.
        :param timesteps: Number of timesteps
        :return: alphas, cumulative product of alphas, and beta
        """
        betas = torch.linspace(0.0001, 0.02, timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas, alphas_cumprod, betas

    def generate_alphas_cosine(self, timesteps, s=0.008):
        """
        Generate alphas using cosine strategy.
        :param timesteps: Number of timesteps
        :return: alphas, cumulative product of alphas, and beta
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)

        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # betas nur für den Sampling-Schritt
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, max=0.999)
        alphas = 1.0 - betas

        alphas_cumprod = alphas_cumprod[1:]

        return alphas, alphas_cumprod, betas

    def generate_noisy_image(self, images, alphas_cumprod, time):
        """
        Generate noisy images from clean images at timestep 'time'.
        :param images: clean images
        :param alphas_cumprod: cumulative alphas parameter
        :param time: timestep for noisy images
        :return: noisy images and noise
        """
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod)

        noise = torch.randn_like(images)
        noisy_image = sqrt_alpha_cumprod[time].view(-1, 1, 1, 1) * images + sqrt_one_minus_alpha_cumprod[time].view(-1, 1, 1, 1) * noise

        return noisy_image, noise

    @torch.no_grad()
    def sample_one_step(self, x, t , labels=None, betti_numbers=None):
        """
        Performing exactly oen diffusion step.
        :param x: noisy images
        :param t: timestep
        :param labels: labels. Unused.
        :param betti_numbers: Betti number constraint of the noisy images x.
        :return: (noisy) images after one diffusion step
        """
        self.eval()

        ts = torch.full((x.shape[0],), t, device=self.device)

        alphas_t = self.alphas[t].view(1, 1, 1, 1)
        betas_t = self.betas[t].view(1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(1, 1, 1, 1)

        noise = self.unet(x, ts, topo_constraints=betti_numbers).sample

        mu = (1 / torch.sqrt(alphas_t)) * (x - (betas_t / torch.sqrt(1 - alpha_cumprod_t)) * noise )

        if t == 0:
            return mu
        else:
            var = betas_t
            return mu + torch.sqrt(var) * torch.randn_like(mu)


    @torch.no_grad()
    def sample(self, num_samples, num_steps=0, labels=None, betti_numbers=None):
        """
        Sample images.
        :param num_samples: number of samples
        :param num_steps: Unused. using global timestep parameter instead. Kept for compatability.
        :param labels: labels. Unused. Kept for compatability.
        :param betti_numbers: The Betti number constraint for reach image.
        :return: sampled images
        """
        self.eval()
        x = torch.randn((num_samples, self.in_channels, self.image_dim, self.image_dim), device=self.device)

        for t in reversed(range(self.timesteps)):
            x = self.sample_one_step(x, t, labels, betti_numbers)

        return x


    @torch.no_grad()
    def sample_ddim(self, num_samples, num_steps=0, eta=0.0, labels=None, betti_numbers=None):
        """
        Sample images using DDIm strategy.
        :param num_samples: number of samples
        :param num_steps: number of sampling (DDIM) steps
        :param eta: eta hyperparameter of DDIM
        :param labels: labels. Unused. Kept for compatability.
        :param betti_numbers: The Betti number constraint for reach image.
        :return: sampled images
        """
        self.eval()
        x = torch.randn((num_samples, self.in_channels, self.image_dim, self.image_dim), device=self.device)

        step_size = self.timesteps // num_steps

        timesteps = list(reversed(range(0, self.timesteps, step_size)))
        for idx, t in enumerate(timesteps):
            t_prev = timesteps[idx+1] if idx + 1 < len(timesteps) else -1

            ts = torch.full((num_samples,), t, device=self.device)

            alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_bar_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)

            eps = self.unet(x, ts, topo_constraints=betti_numbers).sample

            x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )

            noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)

            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps + sigma * noise

        return x


    def sample_improve(self, x, betti_numbers, num_steps=0, eta=0.0):
        """
            sample() method but keeps gradient.
        """
        self.eval()
        step_size = self.timesteps // num_steps

        timesteps = list(reversed(range(0, self.timesteps, step_size)))
        for idx, t in enumerate(timesteps):
            x = self._sample_improve_step(x, betti_numbers, idx, timesteps, eta)

        return x


    def _sample_improve_step(self, x, betti_numbers, step, timesteps, eta=0.0):
        """
         Performs on DDIM diffusion step and keeps gradient.
        """
        t = timesteps[step]
        t_prev = timesteps[step + 1] if step + 1 < len(timesteps) else -1

        ts = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.tensor(1.0,
                                                                                                        device=self.device).view(
            -1, 1, 1, 1)

        eps = self.unet(x, ts, topo_constraints=betti_numbers).sample

        x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        sigma = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)

        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(
            1 - alpha_bar_prev - sigma ** 2) * eps + sigma * noise

        return x


    def _predict_x0_from_xt(self, x_t, betti_numbers, t, num_total_steps):
        """
        Directly predict x0 (clean image) from xt (noisy image) in one step.
        :param x_t: noisy images
        :param betti_numbers: The Betti number constraint for reach image.
        :param t: the current timestep
        :param num_total_steps: Total number of DDIM steps. Unused. Kept for compatability.
        :return: predicted clean images
        """
        ts = torch.full((x_t.shape[0],), t, device=self.device)

        eps = self.unet(x_t, ts, topo_constraints=betti_numbers).sample

        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        return x0_pred.clamp(-1, 1)


    def training_step(self, batch: torch.Tensor, batch_idx):
        """
        Perform one training step
        :param batch: Training batch containing especially: Images and Betti Numbers
        :param batch_idx: Batch index
        :return:
        """

        betti_numbers=None
        if len(batch) == 4:
            images, labels, betti_numbers, _ = batch
        else:
            images, labels = batch

        ts = torch.randint(0,  self.timesteps, (images.shape[0],), requires_grad=False, device=self.device)

        noisy_images, noise = self.generate_noisy_image(images, self.alphas_cumprod, ts)

        betti_numbers = betti_numbers if self.use_topo_constraint else None
        noise_estimate = self.unet(noisy_images, ts, topo_constraints=betti_numbers).sample

        alpha_bar = self.alphas_cumprod[ts].view(-1, 1, 1, 1)
        x0_pred = (noisy_images - torch.sqrt(1 - alpha_bar) * noise_estimate) / torch.sqrt(alpha_bar)

        simple_loss = self.loss_function(noise_estimate, noise)
        loss = simple_loss

        if self.use_topo_loss:
            topo_loss = self.topo_loss(x0_pred.clamp(-1, 1), images)
            loss = loss + self.reg_weight * topo_loss.mean()
            self.log("topo_loss", topo_loss.mean(), prog_bar=True, on_step=True, on_epoch=True)

        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)


        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
