# EEGMirror (ICCV 2025)
This is the official implementation of [EEGMirror: Leveraging EEG data in the wild via Montage-Agnostic Self-Supervision for EEG to Video Decoding](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_EEGMirror_Leveraging_EEG_Data_in_the_Wild_via_Montage-Agnostic_Self-Supervision_ICCV_2025_paper.pdf).

[Xuan-Hao Liu](https://xuanhaoliu.github.io/), [Bao-Liang Lu](https://bcmi.sjtu.edu.cn/home/blu/), [Wei-Long Zheng*](https://weilongzheng.github.io/)    
Shanghai Jiao Tong University

## Overview
We present, **EEGMirror**, a brain decoding framework that reconstructs dynamic visual perception from EEG signals by pretraining on EEG data in the wild: **1) Neural Quantization** EEGMirror converts nonstationary raw EEG signals into robust discrete representations via a neural codebook. **2) Montage-Agnostic Position Embedding (MAPE)** EEGMirror derives each channel's position directly from its scalp proportion instead of a fixed per-channel lookup table, letting a single masked-pretrained EEG encoder flexibly leverage heterogeneous EEG datasets that vary in montages. **3) Multimodal Contrastive Alignment** EEGMirror aligns the pretrained encoder with both high-level semantic (CLIP text) and low-level perceptual (per-frame VAE latent) visual information decoded from EEG, which then guides a fine-tuned inflated Stable Diffusion model to reconstruct the video stimuli.

## Data Preprocessing
We use the public SEED and SEED-DV dataset from [this](https://bcmi.sjtu.edu.cn/home/seed/), and FACED dataset from [this](https://www.synapse.org/Synapse:syn50614194). The EEG data files are arranged as follows:

```
📂 EEGdata
┣ 📂 SEED/Preprocessed_EEG
┃   ┣ 📜 1_20131027.mat
┃   ┣ ...
┃   ┣ 📜 15_20131105.mat

┣ 📂 SEED-DV
┃   ┣ 📂 EEG
┃   ┃   ┗ 📜 sub1.npy
┃   ┃   ┗ ...
┃   ┃   ┗ 📜 sub20.npy
┃   ┣ gt_text_embedding.pt #(1400, 77, 768)
┃   ┣ gt_3fps_frame_latent.pt #(1400, 6, 4, 36, 64)

┣ 📂 /FACED/Processed_data
┃   ┣ 📜 sub000.pkl
┃   ┣ ...
┃   ┣ 📜 sub122.pkl
```

## How to run our code
First cd into the `EEGMirror` project root.
### 0. Environment

```bash
conda create -n EEGmirror python=3.10 -y
conda activate EEGmirror
# install the CUDA build of torch/torchvision/torchaudio matching your driver first, e.g. for CUDA 11.8:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt   # the rest: numpy, scipy, timm, einops
```

### 1. Stage 1 — Neural codebook (VQNSP) pretraining
`--data_specs`: pretraining datasets `name:/path/to/root`.  
`--window_sec`: segment window length.  
`--sampling_rate`: raw EEG signals sampling rate, default 200.  
`--patch_len_sec`: EEG patch length, default 0.2.
```bash
python scripts/run_vqnsp_training.py \
  --data_specs seed:/work1/xuanhao/EEGdata/SEED/Preprocessed_EEG \
               seeddv:/work1/xuanhao/EEGdata/SEED-DV/EEG \
               faced:/work1/xuanhao/EEGdata/FACED/Processed_data \
  --window_sec 4.0 --sampling_rate 200 --patch_len_sec 0.2 \
  --model_size small --codebook_size 1024 --code_dim 32 \
  --batch_size 8 --epochs 50 --lr 5e-4 \
  --output_dir ./output/vqnsp
```

### 2. Stage 2 — Masked brain-encoder pretraining with MAPE
`--vqnsp_ckpt`: checkpoint of the trained codebook in the Stage-1.  
`--random_mask_ratio` / `--channel_mask_ratio` / `--frame_mask_ratio`: each strategy's masking ratio.  
```bash
python scripts/run_pretraining.py \
  --data_specs seed:/work1/xuanhao/EEGdata/SEED/Preprocessed_EEG \
               seeddv:/work1/xuanhao/EEGdata/SEED-DV/EEG \
               faced:/work1/xuanhao/EEGdata/FACED/Processed_data \
  --window_sec 4.0 \
  --vqnsp_ckpt ./output/vqnsp/vqnsp_last.pt --vqnsp_model_size small \
  --codebook_size 1024 --code_dim 32 --model_size small \
  --batch_size 8 --epochs 50 --lr 5e-4 \
  --random_mask_ratio 0.5 --channel_mask_ratio 0.5 --frame_mask_ratio 0.75 \
  --output_dir ./output/pretrain
```

### 3. Stage 3 — Brain modality alignment (per subject)

Assumes `gt_text_embedding.pt` (1400, 77, 768 — y1, CLIP text embedding of each clip's BLIP caption) and `gt_3fps_frame_latent.pt` (1400, 6, 4, 36, 64 — y2, per-frame Stable-Diffusion VAE latent) are already prepared under `/work1/xuanhao/EEGdata/SEED-DV/` (see Data Preprocessing above).

`--sub`: subject ID name of .npy file, e.g., "sub1".  
`--pretrain_ckpt`: pretrianed brain encoder's checkpoint.  
`--seq2seq_depth` / `--seq2seq_heads`: seq2seq model's parameters. 

```bash
python scripts/run_alignment.py \
  --sub sub1 \
  --pretrain_ckpt ./output/pretrain/pretrain_last.pt \
  --encoder_model_size small --codebook_size 1024 \
  --batch_size 8 --epochs 50 --lr 1e-4 \
  --seq2seq_depth 4 --seq2seq_heads 8 \
  --output_dir ./output/align --results_dir ./results
```
Add `--co_train_diffusion --unet_ckpt <dir>` to also co-train a pre-fine-tuned inflated
[Tune-A-Video](https://github.com/showlab/Tune-A-Video) model (a `diffusers`
`save_pretrained()` checkpoint, e.g. `./output/finetuned_t2v_ckpt`); fine-tuning that checkpoint on
SEED-DV's videos + BLIP captions is a separate step (Tune-A-Video's own `train_tuneavideo.py`).
Omitting `--unet_ckpt` falls back to a toy UNet stand-in.

## Citation

If you find this work useful in your research, please consider citing:

```
@inproceedings{liu2025eegmirror,
  title={{EEGMirror}: Leveraging {EEG} data in the wild via montage-agnostic self-supervision for {EEG} to video decoding},
  author={Liu, Xuan-Hao and Lu, Bao-Liang and Zheng, Wei-Long},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={18273--18283},
  year={2025},
  organization={IEEE}
}
```
