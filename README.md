# DC-MPC: Discrete Codebook Model Predictive Control
This repository is the official implementation of [DC-MPC](https://www.aidanscannell.com/dcmpc), 
presented in ["Discrete Codebook World Models for Continuous Control"](https://openreview.net/forum?id=lfRYzd8ady) at ICLR 2025.
[DC-MPC](https://www.aidanscannell.com/dcmpc) is a model-based reinforcement learning algorithm demonstrating the 
strengths of learning a discrete latent space with discrete codebook encodings.

> In reinforcement learning (RL), world models serve as internal simulators, enabling agents to predict environment dynamics and future outcomes in order to make informed decisions. While previous approaches leveraging discrete latent spaces, such as DreamerV3, have demonstrated strong performance in discrete action settings and visual control tasks, their comparative performance in state-based continuous control remains underexplored. In contrast, methods with continuous latent spaces, such as TD-MPC2, have shown notable success in state-based continuous control benchmarks. In this paper, we demonstrate that modelling discrete latent states has benefits over continuous latent states and that discrete codebook encodings are more effective representations for continuous control, compared to alternative encodings, such as one-hot and label-based encodings. Based on these insights, we introduce DCWM: **D**iscrete **C**odebook **W**orld **M**odel, a self-supervised world model with a discrete and stochastic latent space, where latent states are codes from a codebook. We combine DCWM with decision-time planning to get our model-based RL algorithm, named DC-MPC: **D**iscrete **C**odebook **M**odel **P**redictive **C**ontrol, which performs competitively against recent state-of-the-art algorithms, including TD-MPC2 and DreamerV3, on continuous control benchmarks. See our project website [www.aidanscannell.com/dcmpc](https://www.aidanscannell.com/dcmpc).


## Install instructions
Install dependencies:
```sh
conda env create -f environment.yml
conda activate dcmpc
```
You might need to install PyTorch with CUDA/ROCm.

## Running experiments
Train the agent:
``` sh
python train.py env=walker-walk
```
To log metrics with W&B:
``` sh
python train.py env=walker-walk ++use_wandb=True
```
All tested tasks are listed in`cfgs/env`.

## Configuring experiments
This repo uses hydra for configuration.
You can easily try new hyperparameters for `DC-MPC` by overriding them on the command line. For example,
``` sh
python train.py env=walker-walk ++use_wandb=True ++agent.batch_size=1024
```
changes the batch size to be 1024 instead of the default value found in `dcmpc.py/DCMPCConfig`.

You can also use hydra to submit multiple Slurm jobs directly from the command line using
``` sh
python train.py -m env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4
```
This uses `utils/cluster_utils.py/SlurmConfig` to configure the jobs, setting `timeout_min=1440` (i.e. 24hrs) and `mem_gb=32`.
If you want to run the job for longer (e.g 48hrs), you can use the following
``` sh
python train.py -m env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4 ++hydra.launcher.timeout_min=2880
```

# BibTeX
Please consider citing our paper:
``` bibtex
@inproceedings{scannell2025discrete,
  title     = {Discrete Codebook World Models for Continuous Control},
  author    = {Aidan Scannell and Mohammadreza Nakhaeinezhadfard and Kalle Kujanp{\"a}{\"a} and Yi Zhao and Kevin Sebastian Luck and Arno Solin and Joni Pajarinen},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=lfRYzd8ady}
}
```
