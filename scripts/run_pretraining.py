#!/usr/bin/env python
# --------------------------------------------------------
# EEGMirror -- Stage 2: masked pre-training of the brain encoder E_brain
# with MAPE + {random, channel, frame} masking (see
# eegmirror/modeling_pretrain.py and eegmirror/masking.py).
#
# Requires a frozen Stage-1 VQNSP checkpoint to supply the target
# codebook indices for the masked-token-prediction objective. Reads
# unlabeled raw EEG directly from the real datasets via --data_specs,
# same convention as scripts/run_vqnsp_training.py.
# --------------------------------------------------------
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from eegmirror.datasets import build_raw_eeg_datasets, pretrain_collate
from eegmirror.modeling_vqnsp import build_vqnsp
from eegmirror.modeling_pretrain import build_pretrain_model
from eegmirror.utils import AverageMeter, load_checkpoint, save_args, save_checkpoint, set_seed


def get_args():
    p = argparse.ArgumentParser('EEGMirror Stage 2: masked brain-encoder pre-training')
    p.add_argument('--data_specs', type=str, nargs='+', required=True,
                    help="One or more 'name:/path/to/root' specs, name in {seed, seeddv, faced} -- "
                         "see run_vqnsp_training.py --data_specs for the exact format/examples.")
    p.add_argument('--window_sec', type=float, default=4.0)
    p.add_argument('--max_files', type=int, default=0,
                    help='Cap the number of subject/session files read per dataset (0 = use all).')
    p.add_argument('--vqnsp_ckpt', type=str, required=True)
    p.add_argument('--vqnsp_model_size', type=str, default='small', choices=['small', 'base'])
    p.add_argument('--codebook_size', type=int, default=1024)
    p.add_argument('--code_dim', type=int, default=32)
    p.add_argument('--model_size', type=str, default='small', choices=['small', 'base'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--random_mask_ratio', type=float, default=0.5)
    p.add_argument('--channel_mask_ratio', type=float, default=0.5)
    p.add_argument('--frame_mask_ratio', type=float, default=0.75)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir', type=str, default='./output/pretrain')
    p.add_argument('--max_steps_per_epoch', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_args(os.path.join(args.output_dir, 'args.json'), args)

    ckpt = torch.load(args.vqnsp_ckpt, map_location='cpu')
    patch_len = ckpt['patch_len']
    sampling_rate = ckpt['sampling_rate']
    print(f"[cfg] patch_len = {patch_len} samples (loaded from {args.vqnsp_ckpt})")

    vqnsp = build_vqnsp(patch_len=patch_len, size=args.vqnsp_model_size,
                         n_embed=args.codebook_size, code_dim=args.code_dim).to(args.device)
    load_checkpoint(args.vqnsp_ckpt, vqnsp, map_location=args.device)
    vqnsp.eval()
    for p_ in vqnsp.parameters():
        p_.requires_grad_(False)

    raw_datasets = build_raw_eeg_datasets(
        args.data_specs, window_sec=args.window_sec, sampling_rate=sampling_rate,
        max_files=args.max_files or None,
    )
    loaders = []
    for spec, ds in zip(args.data_specs, raw_datasets):
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                         collate_fn=pretrain_collate(patch_len))
        loaders.append((dl, ds.ch_names))
        print(f"[data] {spec}: {len(ds)} windows, {len(ds.ch_names)} channels")

    model = build_pretrain_model(patch_len=patch_len, size=args.model_size,
                                  vocab_size=args.codebook_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    mask_ratios = {'random': args.random_mask_ratio, 'channel': args.channel_mask_ratio,
                   'frame': args.frame_mask_ratio}

    for epoch in range(args.epochs):
        model.train()
        meter = AverageMeter()
        step = 0
        for dl, ch_names in loaders:
            for patches in dl:
                patches = patches.to(args.device)
                with torch.no_grad():
                    target_ids = vqnsp.get_codebook_indices(patches, ch_names)
                loss, log = model(patches, ch_names, target_ids, mask_ratios=mask_ratios)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                meter.update(loss.item())
                step += 1
                if step % 5 == 0 or step == 1:
                    print(f"[epoch {epoch}] step {step}: total={loss.item():.4f} "
                          f"random={log['random_loss'].item():.4f}(acc={log['random_acc'].item():.2f}) "
                          f"channel={log['channel_loss'].item():.4f}(acc={log['channel_acc'].item():.2f}) "
                          f"frame={log['frame_loss'].item():.4f}(acc={log['frame_acc'].item():.2f})")
                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break
        print(f"== epoch {epoch} done, avg total loss = {meter.avg:.4f} ==")
        save_checkpoint(os.path.join(args.output_dir, f'pretrain_epoch{epoch}.pt'), model,
                         optimizer=optimizer, epoch=epoch,
                         extra={'patch_len': patch_len, 'sampling_rate': sampling_rate})

    save_checkpoint(os.path.join(args.output_dir, 'pretrain_last.pt'), model,
                     extra={'patch_len': patch_len, 'sampling_rate': sampling_rate})
    print(f"[done] Stage 2 checkpoint saved to {args.output_dir}/pretrain_last.pt "
          f"(contains the E_brain encoder used by Stage 3)")


if __name__ == '__main__':
    main()
