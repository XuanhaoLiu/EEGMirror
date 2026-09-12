#!/usr/bin/env python
# --------------------------------------------------------
# EEGMirror -- Stage 3 (+ optional Stage 4): fine-tune the pretrained
# E_brain encoder with semantic (SoftCLIP+MSE) and low-level (Seq2Seq MSE)
# alignment on --sub's downstream EEG2Video (x, y1, y2) data (see
# eegmirror/modeling_align.py and eegmirror/datasets.py:EEG2VideoDataset,
# which reads x directly from real SEED-DV EEG and y1/y2 from the shared
# target tensors built by scripts/prepare_eeg2video_targets.py -- no
# per-subject cache directory is read or written). Pass --co_train_diffusion
# to additionally run the Stage-4 inflated-diffusion co-training step
# (eegmirror/modeling_diffusion.py): with --unet_ckpt, this loads the REAL
# fine-tuned Tune-A-Video inflated UNet; without it, a toy stand-in
# (TinyInflatedUNet) is used so the pipeline still runs with no checkpoint.
# --------------------------------------------------------
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from eegmirror.datasets import (
    EEG2VideoDataset, eeg2video_collate,
    SEEDDV_EEG_ROOT_DEFAULT, SEEDDV_TEXT_EMBED_DEFAULT, SEEDDV_FRAME_LATENT_DEFAULT,
)
from eegmirror.modeling_pretrain import build_pretrain_model
from eegmirror.modeling_align import EEGMirrorAlignModel
from eegmirror.modeling_diffusion import DiffusionCoTrainer, TinyInflatedUNet, build_tuneavideo_unet
from eegmirror.utils import AverageMeter, load_checkpoint, save_args, save_checkpoint, set_seed


def get_args():
    p = argparse.ArgumentParser('EEGMirror Stage 3: brain-modality alignment')
    p.add_argument('--sub', type=str, required=True,
                    help="SEED-DV subject id to fit, e.g. 'sub1' (must match <eeg_root>/<sub>.npy).")
    p.add_argument('--eeg_root', type=str, default=SEEDDV_EEG_ROOT_DEFAULT,
                    help='Directory holding SEED-DV EEG/<sub>.npy files.')
    p.add_argument('--text_embed_path', type=str, default=SEEDDV_TEXT_EMBED_DEFAULT,
                    help='(1400, 77, 768) CLIP text embeddings, shared across every subject '
                         '(built by scripts/prepare_eeg2video_targets.py).')
    p.add_argument('--frame_latent_path', type=str, default=SEEDDV_FRAME_LATENT_DEFAULT,
                    help='(1400, 6, 4, 36, 64) per-frame VAE latents, shared across every subject '
                         '(built by scripts/prepare_eeg2video_targets.py).')
    p.add_argument('--n_test', type=int, default=200,
                    help='Number of trailing clips (last SEED-DV block) held out as the test split.')
    p.add_argument('--pretrain_ckpt', type=str, required=True, help='Stage-2 checkpoint (contains E_brain).')
    p.add_argument('--encoder_model_size', type=str, default='small', choices=['small', 'base'])
    p.add_argument('--codebook_size', type=int, default=1024, help='must match Stage-2 vocab_size')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seq2seq_depth', type=int, default=2)
    p.add_argument('--seq2seq_heads', type=int, default=4)
    p.add_argument('--co_train_diffusion', action='store_true',
                    help='Additionally run the Stage-4 inflated-diffusion co-training step.')
    p.add_argument('--unet_ckpt', type=str, default=None,
                    help='Directory holding a fine-tuned Tune-A-Video inflated UNet checkpoint '
                         "(a diffusers-style save_pretrained() output, i.e. <unet_ckpt>/unet/config.json "
                         "+ weights -- e.g. /work1/xuanhao/EEGMirror/output/finetuned_t2v_ckpt), loaded via "
                         'eegmirror/modeling_diffusion.py:build_tuneavideo_unet(). Only used with '
                         '--co_train_diffusion; if omitted, the toy TinyInflatedUNet stand-in is used instead.')
    p.add_argument('--unet_dtype', type=str, default=None, choices=[None, 'fp16', 'fp32', 'bf16'],
                    help='Cast the loaded --unet_ckpt to this dtype (e.g. fp16 to save GPU memory). '
                         'Ignored without --unet_ckpt.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir', type=str, default='./output/align')
    p.add_argument('--results_dir', type=str, default='./results',
                    help='Where test-set predictions (pred_text_embed.pt / pred_frame_latents.pt) are saved, '
                         'under <results_dir>/<sub>/.')
    p.add_argument('--max_steps_per_epoch', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_args(os.path.join(args.output_dir, 'args.json'), args)

    ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
    patch_len = ckpt['patch_len']
    print(f"[cfg] patch_len = {patch_len} samples (loaded from {args.pretrain_ckpt})")

    mem_model = build_pretrain_model(patch_len=patch_len, size=args.encoder_model_size,
                                      vocab_size=args.codebook_size)
    load_checkpoint(args.pretrain_ckpt, mem_model)
    encoder = mem_model.get_encoder().to(args.device)

    ds_kwargs = dict(eeg_root=args.eeg_root, text_embed_path=args.text_embed_path,
                      frame_latent_path=args.frame_latent_path, patch_len=patch_len, n_test=args.n_test)
    train_ds = EEG2VideoDataset(args.sub, split='train', **ds_kwargs)
    test_ds = EEG2VideoDataset(args.sub, split='test', **ds_kwargs)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=eeg2video_collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eeg2video_collate)
    ds = train_ds
    print(f"[data] {args.sub}: {len(train_ds)} train / {len(test_ds)} test samples, {len(ds.ch_names)} channels "
          f"(read directly from {args.eeg_root}/{args.sub}.npy, no cache directory)")

    sample_x, sample_y1, sample_y2 = ds[0]
    num_text_tokens, text_dim = sample_y1.shape
    num_frames, c, h, w = sample_y2.shape
    print(f"[cfg] y1 (semantic target) shape = ({num_text_tokens}, {text_dim})")
    print(f"[cfg] y2 (low-level target) shape = ({num_frames}, {c}, {h}, {w})")

    align_model = EEGMirrorAlignModel(
        encoder, embed_dim=encoder.embed_dim, num_text_tokens=num_text_tokens, text_dim=text_dim,
        num_frames=num_frames, latent_shape=(c, h, w),
        seq2seq_depth=args.seq2seq_depth, seq2seq_heads=args.seq2seq_heads,
    ).to(args.device)

    if args.co_train_diffusion:
        if args.unet_ckpt:
            dtype_map = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
            unet = build_tuneavideo_unet(args.unet_ckpt, torch_dtype=dtype_map.get(args.unet_dtype))
            print(f"[cfg] Stage 4 UNet: real Tune-A-Video checkpoint from {args.unet_ckpt} "
                  f"(use_frame_cond={unet.use_frame_cond})")
        else:
            unet = TinyInflatedUNet(latent_shape=(c, h, w), text_dim=text_dim)
            print("[cfg] Stage 4 UNet: toy TinyInflatedUNet stand-in (pass --unet_ckpt for the real one)")
        unet = unet.to(args.device)
        model = DiffusionCoTrainer(align_model, unet).to(args.device)
    else:
        model = align_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        model.train()
        meter = AverageMeter()
        for step, (x, y1, y2) in enumerate(dl, start=1):
            x, y1, y2 = x.to(args.device), y1.to(args.device), y2.to(args.device)
            if args.co_train_diffusion:
                out = model(x, ds.ch_names, video_latents=y2, text_target=y1, frame_target=y2)
            else:
                out = model(x, ds.ch_names, text_target=y1, frame_target=y2)
            loss = out['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(loss.item())
            if step % 5 == 0 or step == 1:
                loss_str = ' '.join(f'{k}={v.item():.4f}' for k, v in out['losses'].items())
                print(f"[epoch {epoch}] step {step}: total={loss.item():.4f} ({loss_str})")
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
        print(f"== epoch {epoch} done, avg total loss = {meter.avg:.4f} ==")
        save_checkpoint(os.path.join(args.output_dir, f'align_epoch{epoch}.pt'), model, optimizer=optimizer, epoch=epoch)

    save_checkpoint(os.path.join(args.output_dir, 'align_last.pt'), model)
    print(f"[done] Stage 3{'+4' if args.co_train_diffusion else ''} checkpoint saved to {args.output_dir}/align_last.pt")

    # ------------------------------------------------------------------
    # Test-set inference: run the fine-tuned encoder + semantic/low-level
    # heads (align_model, NOT the diffusion co-trainer) over this subject's
    # held-out test clips (last SEED-DV block, --n_test samples) and save
    # the predicted (77, 768) text embeddings and (M, c, h, w) frame
    # latents -- these are exactly the two inputs a frozen, pretrained
    # inflated-diffusion model would need to actually render video (paper
    # Sec 3.3 / Fig. 3(b)); generating video itself is out of scope here.
    # ------------------------------------------------------------------
    align_model.eval()
    pred_text_chunks, pred_frame_chunks = [], []
    with torch.no_grad():
        for x, y1, y2 in test_dl:
            x = x.to(args.device)
            out = align_model(x, ds.ch_names)
            pred_text_chunks.append(out['pred_text_embed'].cpu())
            pred_frame_chunks.append(out['pred_frame_latents'].cpu())
    pred_text_embed = torch.cat(pred_text_chunks, dim=0)
    pred_frame_latents = torch.cat(pred_frame_chunks, dim=0)

    sub_results_dir = os.path.join(args.results_dir, args.sub)
    os.makedirs(sub_results_dir, exist_ok=True)
    torch.save(pred_text_embed, os.path.join(sub_results_dir, 'pred_text_embed.pt'))
    torch.save(pred_frame_latents, os.path.join(sub_results_dir, 'pred_frame_latents.pt'))
    print(f"[done] {args.sub}: saved test-set predictions "
          f"pred_text_embed{tuple(pred_text_embed.shape)} and "
          f"pred_frame_latents{tuple(pred_frame_latents.shape)} -> {sub_results_dir}/")


if __name__ == '__main__':
    main()
