<div align="center">
  <h1>Fine-Tuning Masked Diffusion for Provable Self-Correction</h1>

  <a href="https://arxiv.org/pdf/2505.18456"><img src="https://img.shields.io/badge/ArXiv-Preprint-red" alt="ArXiv badge"></a>

  ![graphical_abstract](./assets/sudoku.gif)
</div>

**PRISM** is a plug-in remasking framework that fine-tunes any pretrained Masked Diffusion Model (MDM) to predict *per-token quality* in a single forward pass, enabling **self-correction** at inference.
It adds a lightweight objective without changing the base MDM architecture.
- **Theory-backed head:** estimates per-token quality \\(q_i \approx \Pr[x_i = y_i \mid y \oplus m_i]\\).  
- **Practical:** simple to fine-tune; Improves over remasking baselines on Sudoku, unconditional text (170M), and code (LLaDA-8B on MBPP) with modest fine-tuning compute.
<a name="getting_started"></a>

🔔 **News**
- [x] **[2025.11.09]** PRISM codebase and Sudoku dataset have been open-sourced.
- [x] **[2025.10.01]** Paper posted on arXiv.

---

## Getting started

To get started, create a conda environment and install the FlashAttention:

```bash
conda env create -f requirements.yaml
conda activate PRISM
pip install flash-attn==2.6.3
```

Create the following directories to store saved models and logs:
```bash
mkdir outputs
mkdir watch_folder
```

If you are pre-training or fine-tuning on the Sudoku dataset (48K), ensure the CSV files in the `./data/` directory are placed in your `${data.cache_dir}/sudoku/`.


You can download the pretrained MDLM checkpoint (OWT) from this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing), provided by the MDLM repository. Place the downloaded files in the `./outputs/checkpoints` directory.

## Training
### Pre-training
We provide pre-training script for Sudoku only:
```bash
./scripts/pretrain_sudoku.sh
```

### Fine-tuning

For PRISM fine-tuning, we provide the following scripts:

- **Sudoku**:
  ```bash
  ./scripts/finetune_sudoku_prism.sh
  ```

- **OWT**:
  ```bash
  ./scripts/finetune_owt_prism.sh
  ```

## Evaluation
We provide evaluation scripts for the fine-tuned module using a static sampler:

- **Sudoku** (Success Rate):
  ```bash
  ./scripts/sample_sudoku_prism.sh
  ```

- **OWT** (MAUVE, Gen PPL, Entropy):
  For faster evaluation, we support multi-node text generation on the OWT dataset:
  ```bash
  ./scripts/sample_owt_prism.sh
  ```
  ### Additional Loop Strategy

  You can enhance the generated texts by applying a loop strategy. Modify the following parameters in the bash scripts to customize the behavior:

  1. **sampling.loop_steps**: Number of loop iterations to perform.
  2. **sampling.num_remask_loop**: Number of tokens to remask during each iteration.

## Baselines

We provide baseline implementations for comparison:

- **Sudoku**:
  - **Token-Critic**: An unofficial training recipe for the [Token-Critic](https://arxiv.org/abs/2209.04439) approach on the Sudoku dataset:
    ```bash
    ./scripts/finetune_sudoku_token_critic.sh
    ```
  - **ReMDM-conf Sampler**: Evaluate using the ReMDM-conf sampler:
    ```bash
    ./scripts/sample_sudoku_remdm-conf.sh
    ```

- **OWT**:
  - **ReMDM-cap Sampler**: Evaluate using the ReMDM-cap sampler:
    ```bash
    ./scripts/sample_owt_remdm.sh
    ```

### Acknowledgements
This repository was built on top of [ReMDM](https://github.com/kuleshov-group/remdm) which was based on [MDLM](https://github.com/kuleshov-group/mdlm) and [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Citation
```
@article{kim2025fine,
  title={Fine-Tuning Masked Diffusion for Provable Self-Correction},
  author={Kim*, Jaeyeon and Kim*, Seunggeun and Lee*, Taekyun and Pan, David Z and Kim, Hyeji and Kakade, Sham and Chen, Sitan},
  journal={arXiv preprint arXiv:2510.01384},
  year={2025}
}
```
