import torch

from src.model.diffusion_lightning import DiffusionModelLightning
from src.model.flowmatching_lightning import FlowMatchingModelLightning
from src.utils.topoloss_simple import TopoLossMSE2D


class GuidedSampler:
    def __init__(self, model, num_guiding_epochs: int = 100, random_ablation=False, guiding_learning_rate: float = 1e-4, guidance_scale: float = 1.0, eta: float = 0.0, constraint=0):
        """
        Guider that enables using test-time guiding on  a given Diffusion or Flow matching model.
        :param model: The trained generative model.
        :param num_guiding_epochs: Number of guiding steps for full-path guiding.
        :param random_ablation: "True" if random ablation (no full-path guiding), else False.
        :param guiding_learning_rate: Learning rate for ful-path guiding.
        :param guidance_scale: guidance scalll hyperparameter for stepwise-guiding (see DPS paper for details).
        :param eta: Eta hyperparameter of DDIM (diffusion only)
        :param constraint: Betti number to condition the model on (e.g. 0 for connected components, or 1 for distinct areas)
        """
        self.model = model
        self.num_guiding_epochs = num_guiding_epochs
        self.random_ablation = random_ablation
        self.learning_rate = guiding_learning_rate

        self.guidance_scale = guidance_scale
        self.eta = eta

        self.loss_func = TopoLossMSE2D(True, True, True, constraint)

    def sample_improve_total(self, batch_size, betti_numbers=None, num_steps=1):
        """
        Perform full-path guiding.
        :param batch_size: Batch size used for sampling
        :param betti_numbers: Instances of Betti number to condition on.
        :param num_steps: Number of guiding steps
        :return: samples images.
        """
        was_train = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        x0 = torch.randn((batch_size, self.model.in_channels, self.model.image_dim, self.model.image_dim),
                         requires_grad=True, device=self.model.device)

        if not self.random_ablation:
            opt = torch.optim.Adam([x0], lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_guiding_epochs)

        print("New sampling batch ...")

        x = x0

        xs_save = []
        losses = []
        for epoch in range(self.num_guiding_epochs):
            # For ablation only: Random sample
            if self.random_ablation:
                with torch.no_grad():
                    x0 = torch.randn((batch_size, self.model.in_channels, self.model.image_dim, self.model.image_dim),
                                     requires_grad=True, device=self.model.device)
                    x = self.model.sample_improve(x0, betti_numbers, num_steps=num_steps)
                    topo_loss = self.loss_func(x, betti_numbers).mean()

            else:
                opt.zero_grad()
                x = self.model.sample_improve(x0, betti_numbers, num_steps=num_steps)
                topo_loss = self.loss_func(x, betti_numbers).mean()

            xs_save.append(x.detach().clone())
            losses.append(topo_loss.item())

            if not self.random_ablation:
                topo_loss.backward()
                torch.nn.utils.clip_grad_norm_([x0], max_norm=1.0)
                opt.step()
                scheduler.step()

        best_index = torch.argmin(torch.tensor(losses)).item()

        for p in self.model.parameters():
            p.requires_grad_(True)
        if was_train:
            self.model.train()

        print(losses)

        return xs_save[best_index]

    def sample_improve_stepwise(self, batch_size, betti_numbers=None, num_steps=1):
        """
        Perform stepwise guiding.
        :param batch_size: Batch size used for sampling
        :param betti_numbers: Instances of Betti number to condition on.
        :param num_steps: Number of guiding steps
        :return: samples images.
        """
        was_train = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        x_t = torch.randn(
            (batch_size, self.model.in_channels, self.model.image_dim, self.model.image_dim),
            device=self.model.device
        )

        if isinstance(self.model, FlowMatchingModelLightning):
            del_t = 1 / num_steps
        elif isinstance(self.model, DiffusionModelLightning):
            step_size = self.model.timesteps // num_steps
            timesteps = list(reversed(range(0, self.model.timesteps, step_size)))
        else:
            raise NotImplementedError

        print("New sampling batch ...")

        for i in range(num_steps):

            x_t = x_t.detach().requires_grad_(True)
            if isinstance(self.model, FlowMatchingModelLightning):
                x0_pred = self.model._predict_x1_from_xt(x_t, betti_numbers, i, num_steps)
            elif isinstance(self.model, DiffusionModelLightning):
                x0_pred = self.model._predict_x0_from_xt(x_t, betti_numbers, timesteps[i], num_steps)

            topo_loss = self.loss_func(x0_pred, betti_numbers).mean()

            grad = torch.autograd.grad(topo_loss, x_t)[0]

            with torch.no_grad():
                if isinstance(self.model, FlowMatchingModelLightning):
                    x_t_next = self.model._sample_improve_step(x_t, betti_numbers, i, del_t)
                else:
                    x_t_next = self.model._sample_improve_step(x_t, betti_numbers, i, timesteps, eta=self.eta)

            step_norm = grad.flatten(1).norm(dim=1).view(-1, *([1] * (grad.dim() - 1)))
            zeta = self.guidance_scale / (step_norm + 1e-8)
            x_t = (x_t_next.detach() - zeta * grad).detach()

            print("loss: ", topo_loss)

        for p in self.model.parameters():
            p.requires_grad_(True)
        if was_train:
            self.model.train()

        return x_t.detach()
