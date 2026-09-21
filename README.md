# Topological-constrained Training and Test-Time Guidance for Generative Models
This repository contains the code for the project "Neuroinformatik" completed during the summer term of 2026 at Ulm University.

This work is based on [TopoDiffusionNet][1], a novel method for generating masks that satisfy certain topological
constraints, i.e. binary masks that have a specific Betti number. The example we focussed on in this project
is generating masks that contain a desired number of distinct objects. In general, such masks can be used as a constraint 
for generating images in a second step, e.g. by using a [ControlNet][2].

In this paper, we extend TopoDiffusionNet so that it works not only with diffusion models but also with flow matching models.
Further we introduce and evaluate two test-time guiding mechanisms, called **full-path** and **stepwise**
guiding. These mechanisms steer the sampling process toward a valid solution rather than only operating
at training time.
Details about the background, introduced methods, as well as our findings can be found in the accompanied paper.

Contributors: **Leon Walcher** ([leon.walcher@uni-ulm.de](mailto:leon.walcher@uni-ulm.de)). 

Please feel free to contact me if you have any questions about the code.


## Reproduce our work

To get access to the **exact** synthetic dataset used in the paper, as well as all trained model checkpoints please contact
me. The following subsections describe how
you can create the synthetic dataset on your own and train all models from scratch.

### Environment settings

All experiments regarding GraphAny++ where conducted on a NVIDIA A100 (80GB) with CUDA version 13.2.
All requirements can be installed for Python 3.12.3 and CUDA 13.2 using pip:

```pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132 ```


### Dataset

In our study we sue a synthetic dataset containing masks that show different geometric 2D object.
You can create the dataset we used by running:

```python src/datasets/shapes_generator.py --samples=1000 --c_min=1 --c_max=10 --size=256 --output=data/shapes_dataset```

We also support other dataset containing binary masks (and also images if no topological function is used). 
Please check ```src/datasets/datasets.py``` to find the correct format.

### Model Training

You can reproduce our model training by running. Do not forget to log in with your [wandb][5] account if you want to track the training process.

To train the model without topological loss run:
```
python main.py --model=<MODEL_TYPE> --dataset="shapes" --data_dir="data" --cache_dir=".cache" --image_size=256 --eval_every_n_steps=50000 \
                                         --train_steps=600000 --pos_encoding_dim=256  --batch_size=16 --topo_dim=0 --no_topo_loss --reg_weight=1e-5 \
                                         --learning_rate=1e-4 --num_test_steps=50 --num_gen_samples=50 --device="cuda" --seed=0
```

Set <MODEL_TYPE> to "Diffusion" or "FlowMatching".

To train a model with topoloss we recommend to start from a pre-trained checkpoint of a model trained **without** the topological loss. 
If you want to reproduce our results use the checkpoint after 550,000 steps.
To train the Diffusion model run:
```
python main.py --model="Diffusion" --dataset="shapes" --data_dir="data" --cache_dir=".cache" --image_size=256 --eval_every_n_steps=2500 \
                                         --train_steps=600000 --pos_encoding_dim=256  --batch_size=16 --topo_dim=0 --reg_weight=1e-5 \
                                         --learning_rate=1e-4 --num_test_steps=50 --num_gen_samples=50 --device="cuda" --seed=0 \
                                         --model_checkpoint=<PATH_TO_CHECKPOINT>
```
To train the Flow Matching model run:
```
python main.py --model="FlowMatching" --dataset="shapes" --data_dir="data" --cache_dir=".cache" --image_size=256 --eval_every_n_steps=2500 \
                                         --train_steps=600000 --pos_encoding_dim=256  --batch_size=16 --topo_dim=0 --reg_weight=1e-3 \
                                         --learning_rate=1e-4 --num_test_steps=50 --num_gen_samples=50 --device="cuda" --seed=0 \
                                         --model_checkpoint=<PATH_TO_CHECKPOINT>
```

### Evaluation

The easiest way to evaluate the model is to create move you trained model checkpoint to
```./ckpts_<MODEL_TYPE>_final/``` and rename it to ```<MODEL_TYPE>_wo_topoloss.ckpt``` if trained without topological loss
and ```<MODEL_TYPE>_with_topoloss.ckpt``` if trained with topological loss.
To execute evaluation set the correct model in ```run.sh``` and execute the file via

```bash run.sh <EVAL_VARIANTE>```

Valid options for <EVAL_VARIANT> are "no_topoloss", "topoloss", "full_guiding", and "stepwise_guiding",

You can also use the direct entry point main.py for evaluation.
Run ```python main.py --help``` to list all avialable parameters.


### External Code

The source code for the topological loss functions in ```src/utils/topoloss_normal.py``` and ```src/utils/topoloss_simple.py```
are based on the original [implementation of TopoDiffusionNet][4] by Gupta et al.


### Acknowledgement

The authors acknowledge support from the state of Baden-Württemberg through bwHPC.

[1]: https://openreview.net/forum?id=ZK1LoTo10R
[2]: https://arxiv.org/abs/2302.05543
[3]: paper.pdf
[4]: https://github.com/Saumya-Gupta-26/TopoDiffusionNet
[5]: https://wandb.ai/