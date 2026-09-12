# --------------------------------------------------------
# EEGMirror -- datasets.
#
# Two families:
#   1) Pre-training datasets (Stage 1 & 2): raw multi-channel EEG only,
#      no labels. `RawEEGDataset` reads directly from the three real
#      "in-the-wild" datasets used in the paper -- SEED, SEED-DV, FACED --
#      cutting each recording into non-overlapping `window_sec`-second
#      windows (default 4s) and resampling everything to a common
#      `sampling_rate` (200Hz; SEED and SEED-DV are already 200Hz, only
#      FACED's native 250Hz needs resampling).
#   2) The downstream EEG2Video dataset (Stage 3 & 4): (x, y1, y2) triples
#         x  : (C, T)         raw EEG segment, e.g. (62, 400) = 2s @ 200Hz
#         y1 : (77, 768)      CLIP text embedding of the BLIP caption
#         y2 : (M, c, h, w)   per-frame Stable-Diffusion VAE latents
#      `EEG2VideoDataset` reads these directly from the real SEED-DV data
#      for a given subject (x sliced on the fly from EEG/<sub>.npy; y1/y2
#      looked up from the two SUBJECT-INDEPENDENT target tensors built by
#      scripts/prepare_eeg2video_targets.py) -- no per-subject cache
#      directory is written to disk.
# --------------------------------------------------------
import os
import pickle
import re
from functools import lru_cache

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .utils import video_latent_shape

# ----------------------------------------------------------------------
# Per-dataset electrode montages, in the exact channel order each
# dataset's arrays store them (needed for MAPE, see channels.py). SEED and
# SEED-DV share the same 62-channel ESI NeuroScan cap used by the SJTU
# BCMI lab; FACED uses a different 32-channel cap. Copied from the
# channel-order table already used elsewhere in this user's codebase
# (/work1/xuanhao/Finetuning_lora/datasets/electrodes.py), which notes
# these were cross-checked against production use but not against a
# channel-order file shipped with the datasets themselves (none ships
# with any of the three) -- see that file's docstring for full provenance.
# ----------------------------------------------------------------------
SEED_ELECTRODES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'FT8', 'T3', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T4', 'TP7',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'T5', 'P5', 'P3',
    'P1', 'PZ', 'P2', 'P4', 'P6', 'T6', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
    'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2',
]
SEEDDV_ELECTRODES = SEED_ELECTRODES  # same lab, same 62-channel cap
FACED_ELECTRODES = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T3', 'T4', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3',
    'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2', 'A2', 'A1',
]
assert len(SEED_ELECTRODES) == 62 and len(FACED_ELECTRODES) == 32


def segment_patches(x: torch.Tensor, patch_len: int):
    """(C, T) -> (C, K, patch_len) via a non-overlapping sliding window,
    K = floor(T / patch_len). Trailing samples that don't fill a whole
    patch are dropped, matching LaBraM/EEGMirror's patching (Sec 3.1.1).
    """
    C, T = x.shape
    K = T // patch_len
    if K == 0:
        raise ValueError(f"patch_len={patch_len} is longer than the signal length T={T}.")
    x = x[:, :K * patch_len]
    return x.reshape(C, K, patch_len)


def pretrain_collate(patch_len):
    """Returns a collate_fn that stacks raw (C, T) samples into a
    (B, C, K, patch_len) patch tensor for a fixed patch length."""
    def _collate(batch):
        patches = [segment_patches(x, patch_len) for x in batch]
        return torch.stack(patches, dim=0)
    return _collate


# ----------------------------------------------------------------------
# Real, unlabeled raw-EEG readers for Stage 1 / Stage 2 pre-training.
#
# Each of the three "in-the-wild" datasets has its own on-disk format and
# native sampling rate; `_resample_to` + one `_load_<dataset>_resampled`
# function per dataset normalize all of that away into a plain tuple of
# (channels, native-order) float32 arrays already at the *target*
# sampling rate, cached per (file, target_sr) with `functools.lru_cache`
# so repeated epochs don't re-read/re-resample the same file from disk.
# ----------------------------------------------------------------------
def _resample_to(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample the last axis of `x` from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return x
    from scipy.signal import resample
    T = x.shape[-1]
    new_T = int(round(T * target_sr / orig_sr))
    return resample(x, new_T, axis=-1).astype(np.float32)


SEED_NATIVE_SR = 200      # SEED's Preprocessed_EEG is already downsampled to 200Hz
SEEDDV_NATIVE_SR = 200    # SEED-DV's EEG/*.npy is already downsampled to 200Hz
FACED_NATIVE_SR = 250     # FACED's Processed_data/*.pkl is at its raw 250Hz

_SEED_TRIAL_KEY_RE = re.compile(r'_eeg(\d+)$')


@lru_cache(maxsize=8)
def _load_seed_session(mat_path):
    """One SEED session .mat -> tuple of 15 (62, T) float32 arrays (T varies
    per trial, ~185-265s @ 200Hz), ordered by trial number 1..15."""
    mat = sio.loadmat(mat_path)
    numbered = []
    for key in mat.keys():
        if key.startswith('__'):
            continue
        m = _SEED_TRIAL_KEY_RE.search(key)
        if m is not None:
            numbered.append((int(m.group(1)), key))
    numbered.sort(key=lambda x: x[0])
    if not numbered:
        raise ValueError(f"No '*_eegN' trial keys found in {mat_path}")
    return tuple(mat[key].astype(np.float32) for _, key in numbered)


@lru_cache(maxsize=8)
def _load_seed_resampled(mat_path, target_sr):
    trials = _load_seed_session(mat_path)
    return tuple(_resample_to(t, SEED_NATIVE_SR, target_sr) for t in trials)


@lru_cache(maxsize=4)
def _load_seeddv_npy(npy_path):
    """One SEED-DV subject .npy -> (7, 62, 104000) float32 array @ 200Hz."""
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[1] != len(SEEDDV_ELECTRODES):
        raise ValueError(f"Expected (blocks, {len(SEEDDV_ELECTRODES)}, T) in {npy_path}, got {arr.shape}")
    return arr


@lru_cache(maxsize=4)
def _load_seeddv_resampled(npy_path, target_sr):
    arr = _resample_to(_load_seeddv_npy(npy_path), SEEDDV_NATIVE_SR, target_sr)
    return tuple(arr[b] for b in range(arr.shape[0]))  # 7 x (62, T)


@lru_cache(maxsize=8)
def _load_faced_pkl(pkl_path):
    """One FACED subject .pkl -> (28, 32, 7500) float32 array @ 250Hz."""
    with open(pkl_path, 'rb') as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != len(FACED_ELECTRODES):
        raise ValueError(f"Expected (trials, {len(FACED_ELECTRODES)}, T) in {pkl_path}, got {arr.shape}")
    return arr


@lru_cache(maxsize=8)
def _load_faced_resampled(pkl_path, target_sr):
    arr = _resample_to(_load_faced_pkl(pkl_path), FACED_NATIVE_SR, target_sr)
    return tuple(arr[t] for t in range(arr.shape[0]))  # 28 x (32, T)


def _list_seed_files(root_dir):
    return sorted(f for f in os.listdir(root_dir) if f.endswith('.mat') and f != 'label.mat')


def _list_seeddv_files(root_dir):
    pat = re.compile(r'^sub\d+(?:_session\d+)?\.npy$')
    return sorted(f for f in os.listdir(root_dir) if pat.match(f))


def _list_faced_files(root_dir):
    return sorted(f for f in os.listdir(root_dir) if f.endswith('.pkl'))


# name -> (per-file loader(file_path, target_sr) -> tuple[(C, T) array, ...],
#          channel-name list, file-listing function)
_RAW_SOURCES = {
    'seed': (_load_seed_resampled, SEED_ELECTRODES, _list_seed_files),
    'seeddv': (_load_seeddv_resampled, SEEDDV_ELECTRODES, _list_seeddv_files),
    'faced': (_load_faced_resampled, FACED_ELECTRODES, _list_faced_files),
}
_DATASET_ALIASES = {'seed-dv': 'seeddv', 'seed_dv': 'seeddv', 'sjtu-seed-dv': 'seeddv'}


class RawEEGDataset(Dataset):
    """Unlabeled raw EEG for Stage 1 / Stage 2 pre-training, read directly
    from one of the three real "in-the-wild" datasets (paper Sec 4.1):

        dataset='seed'    root_dir=.../SEED/Preprocessed_EEG     (62ch,  200Hz native)
        dataset='seeddv'  root_dir=.../SEED-DV/EEG                (62ch,  200Hz native)
        dataset='faced'   root_dir=.../FACED/Processed_data       (32ch,  250Hz native)

    Every recording is cut into non-overlapping `window_sec`-second windows
    (default 4s) after resampling to a common `sampling_rate` (200Hz), so a
    single DataLoader over this dataset yields plain (C, T) float32 tensors,
    T = round(window_sec * sampling_rate), ready for `pretrain_collate`.
    No labels/splits are used -- this is exactly the unlabeled self-
    supervised pre-training data the paper leverages "in the wild".

    `max_files` optionally caps how many subject/session files are read
    (handy for a fast smoke test); leave it at None to use every file for
    a real pre-training run.
    """

    def __init__(self, dataset, root_dir, window_sec=4.0, sampling_rate=200, max_files=None):
        name = _DATASET_ALIASES.get(dataset.lower(), dataset.lower())
        if name not in _RAW_SOURCES:
            raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(_RAW_SOURCES)}")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"{name} root_dir not found: {root_dir}")

        loader_fn, ch_names, list_fn = _RAW_SOURCES[name]
        files = list_fn(root_dir)
        if not files:
            raise FileNotFoundError(f"No {name} files found under {root_dir}")
        if max_files:
            files = files[:max_files]

        self.dataset = name
        self.ch_names = list(ch_names)
        self.sampling_rate = sampling_rate
        self.window_len = int(round(window_sec * sampling_rate))
        self._loader_fn = loader_fn

        self._index = []  # list of (file_path, segment_idx, start_sample)
        for f in files:
            file_path = os.path.join(root_dir, f)
            segments = loader_fn(file_path, sampling_rate)
            for seg_idx, seg in enumerate(segments):
                n_windows = seg.shape[-1] // self.window_len
                for w in range(n_windows):
                    self._index.append((file_path, seg_idx, w * self.window_len))
        if not self._index:
            raise ValueError(
                f"No {window_sec}s windows fit any {name} recording under {root_dir} "
                f"(window_len={self.window_len} samples @ {sampling_rate}Hz)"
            )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_path, seg_idx, start = self._index[idx]
        seg = self._loader_fn(file_path, self.sampling_rate)[seg_idx]
        x = seg[:, start:start + self.window_len]
        return torch.from_numpy(np.ascontiguousarray(x)).float()


def build_raw_eeg_datasets(data_specs, window_sec=4.0, sampling_rate=200, max_files=None):
    """Parse `--data_specs` CLI entries of the form 'name:/path/to/root'
    (e.g. 'seed:/work1/xuanhao/EEGdata/SEED/Preprocessed_EEG') into a list
    of RawEEGDataset instances, one per entry."""
    datasets = []
    for spec in data_specs:
        if ':' not in spec:
            raise ValueError(f"--data_specs entries must look like 'name:/path/to/dir', got {spec!r}")
        name, root_dir = spec.split(':', 1)
        datasets.append(RawEEGDataset(name, root_dir, window_sec=window_sec,
                                       sampling_rate=sampling_rate, max_files=max_files))
    return datasets


# ----------------------------------------------------------------------
# SEED-DV clip epoching, shared by the cache-builder script and the
# EEG2Video downstream dataset below.
#
# Each SEED-DV block (104000 samples @ 200Hz = 520s) presents 40 concepts
# back-to-back; each concept = a 3s text hint followed by 5 non-overlapping
# 2s video clips (3 + 5*2 = 13s/concept * 40 = 520s, matching the block
# length exactly). This yields 40*5 = 200 clips/block * 7 blocks = 1400
# clips/subject, in the same block-major / concept-major / clip-minor
# order as the 1400 lines across BLIP-caption/{1st..7th}_10min.txt.
# ----------------------------------------------------------------------
SEEDDV_NUM_BLOCKS = 7
SEEDDV_CONCEPTS_PER_BLOCK = 40
SEEDDV_CLIPS_PER_CONCEPT = 5
SEEDDV_HINT_SEC = 3.0
SEEDDV_CLIP_SEC = 2.0
SEEDDV_CLIPS_PER_SUBJECT = SEEDDV_NUM_BLOCKS * SEEDDV_CONCEPTS_PER_BLOCK * SEEDDV_CLIPS_PER_CONCEPT  # 1400


def seeddv_clip_eeg(blocks, sampling_rate=200):
    """blocks: (7, C, T) EEG @ sampling_rate (as loaded from EEG/subX.npy).
    Returns a list of 1400 (C, clip_len) arrays, clip_len = round(2s * sampling_rate),
    in block-major/concept-major/clip-minor order (global clip index =
    block*200 + concept*5 + clip)."""
    clip_len = int(round(SEEDDV_CLIP_SEC * sampling_rate))
    hint_len = int(round(SEEDDV_HINT_SEC * sampling_rate))
    concept_len = hint_len + SEEDDV_CLIPS_PER_CONCEPT * clip_len
    clips = []
    for b in range(blocks.shape[0]):
        block = blocks[b]
        for c in range(SEEDDV_CONCEPTS_PER_BLOCK):
            base = c * concept_len + hint_len
            for k in range(SEEDDV_CLIPS_PER_CONCEPT):
                start = base + k * clip_len
                clips.append(np.ascontiguousarray(block[:, start:start + clip_len]))
    return clips


# Default real-data locations for the EEG2Video downstream task (all under
# the SEED-DV release), used as the CLI defaults in scripts/run_alignment.py.
SEEDDV_EEG_ROOT_DEFAULT = '/work1/xuanhao/EEGdata/SEED-DV/EEG'
SEEDDV_TEXT_EMBED_DEFAULT = '/work1/xuanhao/EEGdata/SEED-DV/gt_text_embedding.pt'
SEEDDV_FRAME_LATENT_DEFAULT = '/work1/xuanhao/EEGdata/SEED-DV/gt_3fps_frame_latent.pt'


@lru_cache(maxsize=4)
def _load_seeddv_subject_clips_cached(eeg_root, sub, sampling_rate):
    """Read EEG/<sub>.npy, resample, and epoch into 1400 (62, clip_len) real
    EEG clips (cached so building both the 'train' and 'test' splits of the
    same subject, or re-running Stage 3 in the same process, doesn't re-read
    /re-resample the ~50MB .npy file from disk twice)."""
    npy_path = os.path.join(eeg_root, f'{sub}.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"SEED-DV subject file not found: {npy_path}")
    blocks = _resample_to(_load_seeddv_npy(npy_path), SEEDDV_NATIVE_SR, sampling_rate)
    clips = tuple(seeddv_clip_eeg(blocks, sampling_rate=sampling_rate))
    assert len(clips) == SEEDDV_CLIPS_PER_SUBJECT
    return clips


def load_seeddv_subject_clips(eeg_root, sub, sampling_rate=200):
    """Public wrapper: EEG/<sub>.npy -> list of its 1400 (62, clip_len) real EEG clips."""
    return list(_load_seeddv_subject_clips_cached(eeg_root, sub, sampling_rate))


@lru_cache(maxsize=4)
def _load_shared_target(path):
    """Load one of the two subject-independent alignment targets
    (gt_text_embedding.pt / gt_3fps_frame_latent.pt), cached by path so every
    subject's EEG2VideoDataset reuses the same in-memory tensor instead of
    re-reading a ~300MB file from disk for every subject/split."""
    return torch.load(path, map_location='cpu')


class EEG2VideoDataset(Dataset):
    """Downstream (x, y1, y2) dataset for EEG-to-video alignment /
    generation (Stage 3 & 4) -- reads directly from the real SEED-DV
    recordings, with NO per-subject pre-materialized cache directory:

        x  : (62, 400)        REAL 2s EEG clip @ 200Hz, sliced on the fly
                                out of `<eeg_root>/<sub>.npy` using the
                                3s-hint + 5x2s-clip block epoching (see
                                `seeddv_clip_eeg`).
        y1 : (77, 768)        clip's CLIP text embedding -- identical
                                across every subject (same 1400 clips),
                                loaded once from `text_embed_path`.
        y2 : (6, 4, 36, 64)   clip's per-frame SD-VAE latents -- also
                                identical across every subject, loaded
                                once from `frame_latent_path`.

    `split`/`n_test` slice the 1400 clips (already in block-major /
    concept-major / clip-minor order, see SEEDDV_CLIPS_PER_SUBJECT) to
    match the paper's block-based train/test split (Sec 4.2): with the
    default n_test=200, clip indices 0..1199 (blocks 0-5) are 'train' and
    1200..1399 (block 6) are 'test'.
    """

    def __init__(self, sub, eeg_root=SEEDDV_EEG_ROOT_DEFAULT,
                 text_embed_path=SEEDDV_TEXT_EMBED_DEFAULT,
                 frame_latent_path=SEEDDV_FRAME_LATENT_DEFAULT,
                 split=None, n_test=200, sampling_rate=200, patch_len=None, scale=0.01):
        self.sub = sub
        self.ch_names = list(SEEDDV_ELECTRODES)
        self.patch_len = patch_len
        self.scale = scale

        clips = _load_seeddv_subject_clips_cached(eeg_root, sub, sampling_rate)
        text_embedding = _load_shared_target(text_embed_path)
        frame_latent = _load_shared_target(frame_latent_path)
        n_clips = len(clips)
        if text_embedding.shape[0] != n_clips or frame_latent.shape[0] != n_clips:
            raise ValueError(
                f"Expected {n_clips} rows in both {text_embed_path} ({text_embedding.shape}) and "
                f"{frame_latent_path} ({frame_latent.shape}) to match {sub}'s {n_clips} EEG clips."
            )

        if split == 'train':
            idx_range = range(0, n_clips - n_test) if n_test else range(n_clips)
        elif split == 'test':
            idx_range = range(n_clips - n_test, n_clips) if n_test else range(n_clips)
        elif split is None:
            idx_range = range(n_clips)
        else:
            raise ValueError(f"split must be 'train', 'test', or None, got {split!r}")

        self.indices = list(idx_range)
        self.clips = clips
        self.text_embedding = text_embedding
        self.frame_latent = frame_latent

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gidx = self.indices[i]
        x = torch.from_numpy(np.ascontiguousarray(self.clips[gidx])).float() * self.scale
        if self.patch_len is not None:
            x = segment_patches(x, self.patch_len)  # (C, K, W)
        # .clone() because text_embedding[gidx]/frame_latent[gidx] are views
        # into the whole shared (1400, ...) tensor's storage.
        y1 = self.text_embedding[gidx].float().clone()
        y2 = self.frame_latent[gidx].float().clone()
        return x, y1, y2


def eeg2video_collate(batch):
    xs, y1s, y2s = zip(*batch)
    return torch.stack(xs, 0), torch.stack(y1s, 0), torch.stack(y2s, 0)
