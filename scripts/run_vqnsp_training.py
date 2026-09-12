#!/usr/bin/env python
# --------------------------------------------------------
# EEGMirror -- Stage 1: train the neural codebook (VQNSP), amplitude-only
# reconstruction target (see eegmirror/modeling_vqnsp.py).
#
# Can be pointed at several REAL dataset directories with DIFFERENT
# montages at once (one DataLoader per --data_specs entry) MAPE makes a single shared
# model able to train on all of them. Each entry reads unlabeled raw EEG
# directly off disk in non-overlapping `--window_sec`-second windows (see
# eegmirror/datasets.py:RawEEGDataset)
# --------------------------------------------------------
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from eegmirror.datasets import build_raw_eeg_datasets, pretrain_collate
from eegmirror.modeling_vqnsp import build_vqnsp
from eegmirror.utils import AverageMeter, save_args, save_checkpoint, set_seed


def get_args():
    p = argparse.ArgumentParser('EEGMirror Stage 1: VQNSP (neural codebook) training')
    p.add_argument('--data_specs', type=str, nargs='+', required=True,
                    help="One or more 'name:/path/to/root' specs, name in {seed, seeddv, faced}, e.g. "
                         "'seed:/work1/xuanhao/EEGdata/SEED/Preprocessed_EEG' "
                         "'seeddv:/work1/xuanhao/EEGdata/SEED-DV/EEG' "
                         "'faced:/work1/xuanhao/EEGdata/FACED/Processed_data'. "
                         "Each becomes its own DataLoader (possibly different montages/channel counts).")
    p.add_argument('--window_sec', type=float, default=4.0,
                    help='Non-overlapping raw-EEG window length read from each recording (paper uses 4s here).')
    p.add_argument('--sampling_rate', type=int, default=200,
                    help='Common sampling rate every --data_specs entry is resampled to (SEED/SEED-DV are '
                         'already 200Hz; FACED is resampled from its native 250Hz).')
    p.add_argument('--max_files', type=int, default=0,
                    help='Cap the number of subject/session files read per dataset (0 = use all files; '
                         'set e.g. 1-2 for a fast smoke test).')
    p.add_argument('--patch_len_sec', type=float, default=0.2,
                    help='Sliding-window length in seconds used to cut each --window_sec window into patches. '
                         'LaBraM hard-codes 1s; EEGMirror makes this a tunable hyper-parameter '
                         '(e.g. 0.2s -> 20 patches for a 4s window).')
    p.add_argument('--model_size', type=str, default='small', choices=['small', 'base'])
    p.add_argument('--codebook_size', type=int, default=1024)
    p.add_argument('--code_dim', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir', type=str, default='./output/vqnsp')
    p.add_argument('--max_steps_per_epoch', type=int, default=0, help='0 = full epoch (debug aid).')
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_args(os.path.join(args.output_dir, 'args.json'), args)

    patch_len = round(args.patch_len_sec * args.sampling_rate)
    print(f"[cfg] patch_len = {patch_len} samples ({args.patch_len_sec}s @ {args.sampling_rate}Hz)")

    raw_datasets = build_raw_eeg_datasets(
        args.data_specs, window_sec=args.window_sec, sampling_rate=args.sampling_rate,
        max_files=args.max_files or None,
    )
    loaders = []
    for spec, ds in zip(args.data_specs, raw_datasets):
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                         collate_fn=pretrain_collate(patch_len))
        loaders.append((dl, ds.ch_names))
        print(f"[data] {spec}: {len(ds)} windows, {len(ds.ch_names)} channels")

    model = build_vqnsp(patch_len=patch_len, size=args.model_size,
                         n_embed=args.codebook_size, code_dim=args.code_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        model.train()
        meter = AverageMeter()
        step = 0
        for dl, ch_names in loaders:
            for patches in dl:
                patches = patches.to(args.device)
                loss, log = model(patches, ch_names)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                meter.update(loss.item())
                step += 1
                if step % 5 == 0 or step == 1:
                    print(f"[epoch {epoch}] step {step}: loss={loss.item():.4f} "
                          f"(quant={log['quant_loss'].item():.4f}, "
                          f"rec_amp={log['rec_amplitude_loss'].item():.4f})")
                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break
        print(f"== epoch {epoch} done, avg loss = {meter.avg:.4f} ==")
        save_checkpoint(os.path.join(args.output_dir, f'vqnsp_epoch{epoch}.pt'), model,
                         optimizer=optimizer, epoch=epoch,
                         extra={'patch_len': patch_len, 'sampling_rate': args.sampling_rate})

    save_checkpoint(os.path.join(args.output_dir, 'vqnsp_last.pt'), model,
                     extra={'patch_len': patch_len, 'sampling_rate': args.sampling_rate})
    print(f"[done] Stage 1 checkpoint saved to {args.output_dir}/vqnsp_last.pt")


if __name__ == '__main__':
    main()
