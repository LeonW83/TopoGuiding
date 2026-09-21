import os
import argparse
import shutil
from enum import Enum

import time
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.datasets.datasets import load_dataset
from src.model.diffusion_lightning import DiffusionModelLightning
from src.model.flowmatching_lightning import FlowMatchingModelLightning
from src.ema_callback import EMACallback
from src.model.guided_sampling import GuidedSampler
from src.utils.evaluation import evaluate, evaluate_from_disk
from src.utils.io_handler import save_masks_to_disk, load_masks_from_disk, save_evaluation_metrics

def parse_args():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument("--dataset", type=str, default="shapes", help="Dataset name.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory.")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Path to cache directory.")
    parser.add_argument("--image_size", type=int, default=64, help="Size of the masks.")

    #model
    parser.add_argument("--model", type=str, default="FlowMatching", help="Name of the generative model to use. Either \'FlowMatching\'"
                                                                          " or \'Diffusion\'.")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="Path to checkpoint of pretrained model. \'None\' to use no checkpoint.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save generated images and metrics.")
    parser.add_argument("--results_name", type=str, default=None, help="Folder name where the results are saved in.")


    # model settings
    parser.add_argument("--train_steps", type=int, default=0, help="Number of training steps. Use \'0\' to skip training.")
    parser.add_argument("--eval_every_n_steps", type=int, default=2000, help="Number of training steps after which evaluation is done.")
    parser.add_argument("--train_timesteps", type=int, default=1000, help="Number of timesteps used for training the diffusion model.")
    parser.add_argument("--pos_encoding_dim", type=int, default=256, help="Dimension of the position encoding.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--no_topo_constraint", action="store_false", dest="use_topo_constraint",
                        help="Disable topological constraints during image generation.")
    parser.add_argument("--no_topo_loss", action="store_false", dest="use_topo_loss",
                        help="Disable topological loss during image generation.")
    parser.add_argument("--topo_dim", type=int, default=0,
                        help="Dimension of the topological constraint. \'0\' measures number of connected. \'1\' measures number of holes/areas.")
    parser.add_argument("--reg_weight", type=float, default=1e-3, help="Regularization parameter (weight) of the topological loss.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate of the Adam optimizer.")

    # settings for generation
    parser.add_argument("--num_gen_samples", type=int, default=512, help="Number of images to generate")
    parser.add_argument("--num_test_steps", type=int, default=50, help="Number of steps during image generation.")
    parser.add_argument("--max_betti", type=int, default=10, help="Maximum Betti number to condition on during image generation..")
    parser.add_argument("--guiding_epochs", type=int, default=5, help="Number of guiding epochs during image generation")
    parser.add_argument("--gen_batch_size", type=int, default=1, help="Batch size during image generation")
    parser.add_argument("--guider_lr", type=float, default=1e-3, help="Learning rate for guiding during image generation")
    parser.add_argument("--stepwise_guiding", action="store_true", dest="stepwise_guiding")
    parser.add_argument("--random_ablation", action="store_true", dest="random_ablation")

    # other settings
    parser.add_argument("--device", type=str, default="cuda", help="The device to use. Usually \'cuda\' or \'cpu\'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not args.results_name:
        folder_name = f"results_{args.model}_steps{args.train_steps}"
        if args.guiding_epochs > 0:
            folder_name += "_guiding"
        if args.guiding_epochs > 0 and args.stepwise_guiding:
            folder_name += "_stepwise"

        args.results_path = os.path.join(args.results_dir, folder_name)
    else:
        args.results_path = os.path.join(args.results_dir, args.results_name)

    return args

def load_model(args, data_args):
    """
    Load the generative model (Diffusion/Flow Matching).
    :param args: command line arguments.
    :param data_args: dataset arguments.
    :return: Generative model and ema callback
    """
    if args.model == "Diffusion":
        if args.model_checkpoint is None:
            model = DiffusionModelLightning(data_args.num_channels, data_args.image_size, args.pos_encoding_dim, data_args.num_classes, args.train_timesteps,
                                            learning_rate=args.learning_rate, use_topo_constraint=args.use_topo_constraint or args.use_topo_loss,
                                            use_topo_loss=args.use_topo_loss, topo_dim=args.topo_dim, reg_weight=args.reg_weight).to(args.device)
            ema = None
        else:
            model = DiffusionModelLightning.load_from_checkpoint(checkpoint_path=args.model_checkpoint, in_channels=data_args.num_channels, image_dim=data_args.image_size,
                                                                 use_topo_constraint=args.use_topo_constraint or args.use_topo_loss, use_topo_loss=args.use_topo_loss,
                                                                 learning_rate=args.learning_rate, topo_dim=args.topo_dim, reg_weight=args.reg_weight).to(args.device)
            ckpt = torch.load(args.model_checkpoint, map_location=args.device)
            ema = EMACallback(decay=0.9999)
            ema.on_load_checkpoint(trainer=None, pl_module=model, checkpoint=ckpt)

    elif args.model == "FlowMatching":
        if args.model_checkpoint is None:
            model = FlowMatchingModelLightning(data_args.num_channels, data_args.image_size, args.pos_encoding_dim, data_args.num_classes,
                                               learning_rate=args.learning_rate, use_topo_constraint=args.use_topo_constraint or args.use_topo_loss,
                                               use_topo_loss=args.use_topo_loss, topo_dim=args.topo_dim, reg_weight=args.reg_weight).to(args.device)
            ema = None
        else:
            model = FlowMatchingModelLightning.load_from_checkpoint(checkpoint_path=args.model_checkpoint, in_channels=data_args.num_channels, image_dim=data_args.image_size,
                                                                    use_topo_constraint=args.use_topo_constraint or args.use_topo_loss, use_topo_loss=args.use_topo_loss,
                                                                    learning_rate=args.learning_rate, topo_dim=args.topo_dim, reg_weight=args.reg_weight).to(args.device)
            ckpt = torch.load(args.model_checkpoint, map_location=args.device)
            ema = EMACallback(decay=0.9999)
            ema.on_load_checkpoint(trainer=None, pl_module=model, checkpoint=ckpt)

    else:
        raise NotImplementedError(f"Model type \'{args.model}\' is unknown.")

    return model, ema


def train(model, train_dataset, args, ema):
    """
    Train the generative model
    :param model: The generative model to train.
    :param train_dataset: The training dataset.
    :param args: command line arguments.
    :param ema: The ema callback if exists otherwise None.
    :return: Ema callback after training.
    """

    if args.train_steps <= 0:
        print("Skipped training.")
        return

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True
    )

    ema_callback = ema if ema is not None else EMACallback(decay=0.9999)

    checkpoint_callback = ModelCheckpoint(
        filename="shapes_{step}_wo_topoloss",
        every_n_train_steps=args.eval_every_n_steps,
        save_top_k=-1,
        save_last=True,
    )

    temp_save_path = os.path.join(args.results_dir, f"tmp_{time.time()}" )
    eval_callback = GenerativeEvaluationCallback(
        every_n_steps=args.eval_every_n_steps,
        ema_callback=ema_callback,
        max_eval_betti=args.max_betti,
        eval_batch_size=args.batch_size,
        num_eval_samples=args.num_gen_samples,
        num_gen_steps_eval=args.num_test_steps,
        eval_image_path=temp_save_path
    )

    logger = WandbLogger(
        project="topodiffusion",
        name=f"{args.dataset}_{args.model}",
        save_dir=os.path.join(args.cache_dir, "wandb"),
    )

    accelerator = 'gpu' if args.device == 'cuda' else 'cpu'
    trainer = pl.Trainer(
        callbacks=[ema_callback, checkpoint_callback, eval_callback],
        default_root_dir=args.cache_dir,
        logger=logger,
        max_steps=args.train_steps,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
    )
    ckpt_path = args.model_checkpoint if args.model_checkpoint else None
    if ckpt_path is not None:
        print("[INFO] Using pre-trained checkpoint for training.")
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)

    try:
        shutil.rmtree(temp_save_path)
    except FileNotFoundError:
        print("[INFO] Temporary directory not found. Maybe training as incomplete or evaluatio disabled. Please check training logs.")

    return ema_callback


def generate(model, args, ema):
    """
    Sample masks using a trained generative model
    :param model: A trained generative models.
    :param args: Command line arguments.
    :param ema: Ema callback. Ema is used during sampling.
    :return: The generation time. Generated masks are directly written to disk.
    """
    if ema is not None:
        print("[INFO] Using ema checkpoint for inference.")
        ema._swap_to_ema(model)

    model.to(args.device)
    if args.guiding_epochs > 0:
        guider = GuidedSampler(model, args.guiding_epochs, random_ablation=args.random_ablation, guiding_learning_rate=args.guider_lr, eta=0.0, constraint=args.topo_dim)

    betti_numbers = torch.repeat_interleave(
        torch.arange(1, args.max_betti + 1, device=args.device),
        args.num_gen_samples
    )

    betti_numbers = betti_numbers[torch.randperm(len(betti_numbers))]

    gen_start = time.time()

    append_images = False
    for i in range(0, len(betti_numbers), args.gen_batch_size):
        batch_betti = betti_numbers[i: i + args.gen_batch_size]
        tmp_batch_size = len(batch_betti)

        if args.guiding_epochs > 0:
            if args.stepwise_guiding:
                batch = guider.sample_improve_stepwise(
                    tmp_batch_size,
                    betti_numbers=batch_betti,
                    num_steps=args.num_test_steps
                )
            else:
                batch = guider.sample_improve_total(
                    tmp_batch_size,
                    betti_numbers=batch_betti,
                    num_steps=args.num_test_steps
                )
        else:
            batch = model.sample_ddim(
                tmp_batch_size,
                args.num_test_steps,
                eta=0.0,
                betti_numbers=batch_betti
            )

        save_masks_to_disk(
            batch,
            batch_betti,
            path=args.results_path,
            append_to_existing=append_images
        )
        append_images = True

    generation_time = time.time() - gen_start

    return generation_time


def evaluation(args, generation_time):
    """
    Calculate statistics and saves results to disk.
    :param args: The training arguments
    :param generation_time: The generation time
    :return: Nothing.
    """
    metrics = evaluate_from_disk(path=args.results_path, topo_dim=args.topo_dim, batch_size=args.gen_batch_size)
    metrics["generation_time"] = generation_time
    save_evaluation_metrics(metrics, args.results_path)
    print("Evaluation results: ", metrics)


class GenerativeEvaluationCallback(pl.Callback):

    def __init__(self, every_n_steps=200, ema_callback=None, eval_batch_size=16, num_eval_samples=256, max_eval_betti=10, num_gen_steps_eval=50,
                 eval_image_path = "./results/shapes/"):
        """
        Evaluation callback. Performs evaluation every num_eval_steps steps by sampling images and calculating statistics.
        :param every_n_steps: After how much steps evaluation is done
        :param ema_callback: Reference to the ema callback. Updated dynamically during model training.
        :param eval_batch_size: The evaluation batch size
        :param num_eval_samples: The number of samples to generate per instance of the betti number.
        :param max_eval_betti: Maximum instance of the betti number.
        :param num_gen_steps_eval: Number of DDIM/generation steps used during sampling.
        :param eval_image_path: Temporary path to save the masks during evaluation.
        """
        self.every_n_steps = every_n_steps
        self.ema_callback = ema_callback

        self.eval_batch_size = eval_batch_size
        self.num_eval_samples = num_eval_samples
        self.max_eval_betti = max_eval_betti
        self.num_gen_steps_eval = num_gen_steps_eval
        self.eval_image_path = eval_image_path


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return

        if trainer.global_step % self.every_n_steps == 0:
            self.evaluate(pl_module)

    def evaluate(self, pl_module):
        was_train = pl_module.training
        pl_module.eval()

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        # set only seed 0 for evaluation without changing global RNG =>
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        try:
            if self.ema_callback is not None:
                self.ema_callback._swap_to_ema(pl_module)

            betti_numbers = torch.repeat_interleave(
                torch.arange(1, self.max_eval_betti + 1, device=pl_module.device),
                self.num_eval_samples
            )

            betti_numbers = betti_numbers[torch.randperm(len(betti_numbers))]

            append_images = False
            for i in range(0, len(betti_numbers), self.eval_batch_size):
                batch_betti = betti_numbers[i: i + self.eval_batch_size]
                tmp_batch_size = len(batch_betti)

                batch = pl_module.sample_ddim(
                    tmp_batch_size,
                    self.num_gen_steps_eval,
                    eta=0.0,
                    betti_numbers=batch_betti
                )

                save_masks_to_disk(
                    batch,
                    batch_betti,
                    path=self.eval_image_path,
                    append_to_existing=append_images
                )
                append_images = True

            batch_load, labels = load_masks_from_disk(path=self.eval_image_path)
            metrics = evaluate(batch_load, labels, topo_dim=pl_module.topo_dim)

            for name, value in metrics.items():
                pl_module.log(
                    f"val_{name}",
                    value,
                    prog_bar=True,
                    logger=True,
                )
        finally:
            # Restore old RNG state
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        if self.ema_callback is not None:
            self.ema_callback._swap_to_train(pl_module)

        if was_train:
            pl_module.train()

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    train_dataset, data_args = load_dataset(args.dataset, args.data_dir, image_size=args.image_size)
    model, ema = load_model(args, data_args)
    ema_train = train(model, train_dataset, args, ema)
    ema = ema_train if ema_train is not None else ema
    pl.seed_everything(args.seed)
    gen_time = generate(model, args, ema)
    evaluation(args, gen_time)

if __name__ == '__main__':
    main()