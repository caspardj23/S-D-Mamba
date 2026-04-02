#!/usr/bin/env python3
"""
Prepare per-frame phoneme labels for the Haskins EMA dataset.

Reads TextGrid files (one per sentence) from the original Haskins HPRC corpus
and produces a single .npz file with per-frame phoneme IDs aligned to the
EMA CSV rows.

Usage:
    python scripts/prepare_haskins_phoneme_labels.py \
        --ema_csv    dataset/haskins/ema_8_pos_xz.csv \
        --textgrid_dir  /path/to/haskins_textgrids/ \
        --output     dataset/haskins/phoneme_labels.npz

TextGrid directory structure (expected):
    textgrid_dir/
      F01/
        sentence_0000.TextGrid
        sentence_0001.TextGrid
        ...
      F02/
        ...

Each TextGrid should have a tier named 'phones' (or 'phonemes') with
interval annotations. The script maps each EMA frame (at 100Hz) to the
phoneme interval it falls within.

If you don't have TextGrids yet:
  1. Download the original Haskins HPRC data (includes audio .wav files)
  2. Run Montreal Forced Aligner (MFA) to generate TextGrid alignments:
       mfa align /path/to/haskins_audio/ english_mfa english_mfa /path/to/output_textgrids/
  3. Then run this script to convert TextGrids → per-frame labels

Alternative: If you have a simple CSV with columns (sentence_id, start_time, end_time, phoneme),
use --label_csv instead of --textgrid_dir.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict


def parse_textgrid(textgrid_path):
    """
    Parse a Praat TextGrid file and extract phone intervals.

    Returns list of (start_time, end_time, phone_label) tuples.
    Tries multiple common tier names.
    """
    try:
        # Try praatio first (most robust)
        from praatio import textgrid as tgio
        tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=True)

        # Try common tier names
        for tier_name in ["phones", "phonemes", "phone", "phoneme", "PHN"]:
            if tier_name in tg.tierNames:
                tier = tg.getTier(tier_name)
                return [(start, end, label) for start, end, label in tier.entries]

        # Fall back to first interval tier
        for tier_name in tg.tierNames:
            tier = tg.getTier(tier_name)
            if hasattr(tier, "entries") and len(tier.entries) > 0:
                entry = tier.entries[0]
                if len(entry) == 3:  # interval tier
                    return [(s, e, l) for s, e, l in tier.entries]

        print(f"  WARNING: No phone tier found in {textgrid_path}")
        return []

    except ImportError:
        # Fall back to manual TextGrid parsing
        return _parse_textgrid_manual(textgrid_path)


def _parse_textgrid_manual(textgrid_path):
    """Simple manual TextGrid parser (handles standard Praat format)."""
    with open(textgrid_path, "r") as f:
        lines = f.readlines()

    intervals = []
    i = 0
    in_phone_tier = False

    while i < len(lines):
        line = lines[i].strip()

        # Detect phone tier
        if 'name = "phones"' in line.lower() or 'name = "phonemes"' in line.lower():
            in_phone_tier = True

        # Parse intervals within the phone tier
        if in_phone_tier and "xmin = " in line:
            xmin = float(line.split("=")[1].strip())
            i += 1
            xmax = float(lines[i].strip().split("=")[1].strip())
            i += 1
            text = lines[i].strip().split("=")[1].strip().strip('"')
            intervals.append((xmin, xmax, text))

        # Detect end of tier
        if in_phone_tier and intervals and ("item [" in line and "item [1]" not in line):
            break

        i += 1

    return intervals


def align_labels_to_frames(intervals, n_frames, sample_rate=100):
    """
    Assign a phoneme label to each EMA frame based on time alignment.

    Args:
        intervals: list of (start_time, end_time, phone_label)
        n_frames: number of EMA frames in this sentence
        sample_rate: EMA sampling rate in Hz (100 for Haskins)

    Returns:
        list of phone labels, one per frame
    """
    labels = []
    for frame_idx in range(n_frames):
        t = frame_idx / sample_rate  # frame center time
        label = ""  # default: silence/empty
        for start, end, phone in intervals:
            if start <= t < end:
                label = phone
                break
        labels.append(label)
    return labels


def from_textgrids(ema_csv_path, textgrid_dir, output_path, sample_rate=100):
    """Build phoneme labels from TextGrid files."""
    df = pd.read_csv(ema_csv_path)
    sentence_ids = df["sentence_id"].values
    speaker_ids = df["speaker_id"].values

    # Build sentence table
    sentences = {}
    for sid in np.unique(sentence_ids):
        rows = np.where(sentence_ids == sid)[0]
        sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]])

    # Collect all unique phoneme labels
    all_phonemes = set()
    per_frame_phones = [""] * len(df)
    matched = 0
    missing = 0

    for sid, (start, end, speaker) in sorted(sentences.items()):
        n_frames = end - start

        # Try to find TextGrid file
        tg_path = None
        for pattern in [
            os.path.join(textgrid_dir, speaker, f"sentence_{sid:04d}.TextGrid"),
            os.path.join(textgrid_dir, speaker, f"sentence_{sid}.TextGrid"),
            os.path.join(textgrid_dir, speaker, f"{sid}.TextGrid"),
            os.path.join(textgrid_dir, f"{speaker}_{sid:04d}.TextGrid"),
            os.path.join(textgrid_dir, f"{speaker}_{sid}.TextGrid"),
        ]:
            if os.path.exists(pattern):
                tg_path = pattern
                break

        if tg_path is None:
            missing += 1
            # Fill with empty labels (will map to silence)
            continue

        intervals = parse_textgrid(tg_path)
        if not intervals:
            missing += 1
            continue

        frame_labels = align_labels_to_frames(intervals, n_frames, sample_rate)
        for i, label in enumerate(frame_labels):
            per_frame_phones[start + i] = label
            all_phonemes.add(label)

        matched += 1

    print(f"Matched {matched}/{len(sentences)} sentences with TextGrids")
    if missing > 0:
        print(f"Missing TextGrids for {missing} sentences (labeled as silence)")

    # Build phoneme vocabulary
    phoneme_names = sorted(all_phonemes)
    if "" not in phoneme_names:
        phoneme_names = [""] + phoneme_names
    phone_to_idx = {p: i for i, p in enumerate(phoneme_names)}

    phoneme_ids = np.array(
        [phone_to_idx[p] for p in per_frame_phones], dtype=np.int64
    )

    # Print statistics
    print(f"\nPhoneme vocabulary ({len(phoneme_names)} phones):")
    for pid, pname in enumerate(phoneme_names):
        count = (phoneme_ids == pid).sum()
        pct = count / len(phoneme_ids) * 100
        if count > 0:
            print(f"  {pid:3d}: '{pname:>4s}' — {count:>8d} frames ({pct:.1f}%)")

    # Save
    np.savez(
        output_path,
        phoneme_ids=phoneme_ids,
        phoneme_names=np.array(phoneme_names),
    )
    print(f"\nSaved {len(phoneme_ids)} frame labels to {output_path}")


def from_label_csv(ema_csv_path, label_csv_path, output_path, sample_rate=100):
    """
    Build phoneme labels from a simple CSV with columns:
    sentence_id, start_time, end_time, phoneme
    """
    df_ema = pd.read_csv(ema_csv_path)
    df_labels = pd.read_csv(label_csv_path)

    sentence_ids = df_ema["sentence_id"].values
    n_total = len(df_ema)

    # Build sentence table
    sentences = {}
    for sid in np.unique(sentence_ids):
        rows = np.where(sentence_ids == sid)[0]
        sentences[sid] = (rows[0], rows[-1] + 1)

    per_frame_phones = [""] * n_total

    for _, row in df_labels.iterrows():
        sid = row["sentence_id"]
        if sid not in sentences:
            continue
        start_row, end_row = sentences[sid]
        n_frames = end_row - start_row

        t_start = row["start_time"]
        t_end = row["end_time"]
        phone = row["phoneme"]

        # Convert time to frame indices
        f_start = max(0, int(t_start * sample_rate))
        f_end = min(n_frames, int(t_end * sample_rate))

        for f in range(f_start, f_end):
            per_frame_phones[start_row + f] = phone

    # Build vocabulary and save
    all_phonemes = sorted(set(per_frame_phones))
    if "" not in all_phonemes:
        all_phonemes = [""] + all_phonemes
    phone_to_idx = {p: i for i, p in enumerate(all_phonemes)}
    phoneme_ids = np.array([phone_to_idx[p] for p in per_frame_phones], dtype=np.int64)

    np.savez(
        output_path,
        phoneme_ids=phoneme_ids,
        phoneme_names=np.array(all_phonemes),
    )
    print(f"Saved {len(phoneme_ids)} frame labels ({len(all_phonemes)} phones) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare phoneme labels for Haskins EMA")
    parser.add_argument(
        "--ema_csv", type=str, required=True,
        help="Path to the EMA CSV file (e.g., dataset/haskins/ema_8_pos_xz.csv)"
    )
    parser.add_argument(
        "--textgrid_dir", type=str, default=None,
        help="Directory containing TextGrid files organized by speaker"
    )
    parser.add_argument(
        "--label_csv", type=str, default=None,
        help="Alternative: CSV with columns (sentence_id, start_time, end_time, phoneme)"
    )
    parser.add_argument(
        "--output", type=str, default="dataset/haskins/phoneme_labels.npz",
        help="Output .npz file path"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=100,
        help="EMA sampling rate in Hz (default: 100)"
    )
    args = parser.parse_args()

    if args.textgrid_dir:
        from_textgrids(args.ema_csv, args.textgrid_dir, args.output, args.sample_rate)
    elif args.label_csv:
        from_label_csv(args.ema_csv, args.label_csv, args.output, args.sample_rate)
    else:
        print("ERROR: Provide either --textgrid_dir or --label_csv")
        sys.exit(1)
