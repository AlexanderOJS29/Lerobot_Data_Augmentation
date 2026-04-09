"""Trajectory mirroring tool for LeRobot v3 datasets.

For each episode, creates a horizontally-flipped copy with negated action dimensions.
The output dataset = original episodes + mirrored copies concatenated.
"""

import argparse
import json
import os
import re
import shutil
import tempfile

import cv2
import pandas as pd

from dataset_io import (
    download_dataset,
    copy_non_data_files,
    upload_dataset,
    update_info_json,
)


def _episode_frame_ranges(src_dir: str, episode_indices: list[int]) -> dict[int, tuple[int, int]]:
    """Return {episode_index: (global_start_frame, global_end_frame)} from parquet row positions."""
    # Prefer episodes metadata if it has precomputed dataset_from/to_index
    ep_meta = os.path.join(src_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    if os.path.exists(ep_meta):
        ep_df = pd.read_parquet(ep_meta)
        if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
            ranges = {}
            for _, row in ep_df.iterrows():
                idx = int(row["episode_index"])
                if idx in episode_indices:
                    ranges[idx] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
            if ranges:
                return ranges

    # Fallback: derive from row positions in the data parquet
    data_pq = os.path.join(src_dir, "data", "chunk-000", "file-000.parquet")
    if os.path.exists(data_pq):
        df = pd.read_parquet(data_pq, columns=["episode_index"])
        ranges = {}
        for ep_idx in episode_indices:
            mask = df["episode_index"] == ep_idx
            rows = df.index[mask]
            if len(rows):
                ranges[ep_idx] = (int(rows[0]), int(rows[-1]) + 1)
        return ranges

    return {}


def _open_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("avc1 unavailable")
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


def _read_frames(cap: cv2.VideoCapture, start: int, end: int) -> list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def process_chunked_video(
    input_path: str,
    output_path: str,
    frame_ranges: list[tuple[int, int]],
) -> tuple[int, int]:
    """Write one output chunk: orig frames for all episodes, then mirrored frames.

    Returns (orig_frames_written, mirror_frames_written).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)

    orig_count = 0
    mirror_count = 0

    # Pass 1: original frames in episode order
    for start, end in frame_ranges:
        for frame in _read_frames(cap, start, end):
            writer.write(frame)
            orig_count += 1

    # Pass 2: mirrored frames in the same episode order
    for start, end in frame_ranges:
        for frame in _read_frames(cap, start, end):
            writer.write(cv2.flip(frame, 1))
            mirror_count += 1

    cap.release()
    writer.release()
    return orig_count, mirror_count


def process_per_episode_video(input_path: str, orig_out: str, mirror_out: str) -> int:
    """Copy original file, write a flipped copy. Returns frame count of mirror."""
    shutil.copy2(input_path, orig_out)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(mirror_out, fps, width, height)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(cv2.flip(frame, 1))
        count += 1

    cap.release()
    writer.release()
    return count


def write_episodes_metadata(src_dir: str, out_dir: str,
                            episodes_to_process: list[int], total_original: int) -> None:
    """Copy and extend meta/episodes/chunk-000/file-000.parquet for the mirrored dataset."""
    ep_pq = os.path.join(src_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    if not os.path.exists(ep_pq):
        return

    ep_df = pd.read_parquet(ep_pq)
    orig_rows = ep_df[ep_df["episode_index"].isin(episodes_to_process)].copy().reset_index(drop=True)
    if len(orig_rows) == 0:
        return

    # Remap dataset frame indices to be contiguous from 0
    lengths = orig_rows["length"].tolist() if "length" in orig_rows.columns else []
    if lengths and "dataset_from_index" in orig_rows.columns:
        cumulative = 0
        for i, length in enumerate(lengths):
            orig_rows.at[i, "dataset_from_index"] = cumulative
            orig_rows.at[i, "dataset_to_index"] = cumulative + length
            cumulative += length
        total_orig_frames = cumulative
    else:
        total_orig_frames = int(orig_rows["dataset_to_index"].max()) if "dataset_to_index" in orig_rows.columns else 0

    # Build mirror rows
    mirror_rows = orig_rows.copy()
    mirror_rows["episode_index"] = mirror_rows["episode_index"] + total_original

    if "dataset_from_index" in mirror_rows.columns:
        mirror_rows["dataset_from_index"] = mirror_rows["dataset_from_index"] + total_orig_frames
        mirror_rows["dataset_to_index"] = mirror_rows["dataset_to_index"] + total_orig_frames

    # Offset video timestamps by total duration of original processed episodes
    ts_to_cols = [c for c in mirror_rows.columns if c.endswith("/to_timestamp")]
    for to_col in ts_to_cols:
        from_col = to_col.replace("/to_timestamp", "/from_timestamp")
        total_duration = orig_rows[to_col].max() if to_col in orig_rows.columns else 0.0
        if from_col in mirror_rows.columns:
            mirror_rows[from_col] = mirror_rows[from_col] + total_duration
        mirror_rows[to_col] = mirror_rows[to_col] + total_duration

    combined = pd.concat([orig_rows, mirror_rows], ignore_index=True)

    out_path = os.path.join(out_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"  Episodes metadata written: {len(orig_rows)} orig + {len(mirror_rows)} mirrored episodes")


def mirror_parquet(df: pd.DataFrame, mirror_dims: list[int], episode_offset: int) -> pd.DataFrame:
    """Return a mirrored copy of df with negated action dims and remapped episode indices."""
    mirrored = df.copy()
    if "action" in mirrored.columns:
        def negate_dims(action):
            a = list(action)
            for dim in mirror_dims:
                if dim < len(a):
                    a[dim] = -a[dim]
            return a
        mirrored["action"] = mirrored["action"].apply(negate_dims)
    if "episode_index" in mirrored.columns:
        mirrored["episode_index"] = mirrored["episode_index"] + episode_offset
    return mirrored


def main():
    parser = argparse.ArgumentParser(description="Mirror LeRobot v3 dataset trajectories.")
    parser.add_argument("--source", required=True, help="HF repo id of source dataset")
    parser.add_argument("--output", required=True, help="HF repo id for output dataset")
    parser.add_argument("--mirror-dims", default="0,2",
                        help="Comma-separated action dimension indices to negate (default: 0,2)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Only process first N episodes (for testing)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HF upload, save locally only")
    parser.add_argument("--work-dir", default=None,
                        help="Custom working directory (default: system temp)")
    args = parser.parse_args()

    mirror_dims = [int(d.strip()) for d in args.mirror_dims.split(",") if d.strip()]

    # --- Working directories ---
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        work_root = args.work_dir
    else:
        work_root = tempfile.mkdtemp(prefix="mirror_dataset_")

    src_dir = os.path.join(work_root, "source")
    out_dir = os.path.join(work_root, "output")

    # --- 1. Download ---
    download_dataset(args.source, src_dir)

    # --- 2. Read info.json ---
    info_path = os.path.join(src_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    total_original = info.get("total_episodes", 0)
    n = min(args.episodes, total_original) if args.episodes else total_original
    episodes_to_process = list(range(n))
    print(f"Source has {total_original} episode(s); processing {n}.")

    # --- 3. Parquet: original + mirrored ---
    data_src = os.path.join(src_dir, "data")
    data_dst = os.path.join(out_dir, "data")

    for root, _, files in os.walk(data_src):
        for fname in sorted(files):
            if not fname.endswith(".parquet"):
                continue
            src_pq = os.path.join(root, fname)
            rel = os.path.relpath(src_pq, data_src)
            dst_pq = os.path.join(data_dst, rel)

            df = pd.read_parquet(src_pq)

            if args.episodes is not None and "episode_index" in df.columns:
                df = df[df["episode_index"].isin(episodes_to_process)].reset_index(drop=True)

            mirrored = mirror_parquet(df, mirror_dims, episode_offset=total_original)
            combined = pd.concat([df, mirrored], ignore_index=True)

            os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
            combined.to_parquet(dst_pq, index=False)
            print(f"  Parquet written ({len(df)} orig + {len(mirrored)} mirrored rows): {rel}")

    # --- 4. Videos ---
    video_src = os.path.join(src_dir, "videos")
    video_dst = os.path.join(out_dir, "videos")

    if os.path.isdir(video_src):
        # Compute frame ranges once (used for chunked files)
        frame_ranges_map = _episode_frame_ranges(src_dir, episodes_to_process)
        ordered_ranges = [frame_ranges_map[ep] for ep in episodes_to_process if ep in frame_ranges_map]

        for root, _, files in os.walk(video_src):
            for fname in sorted(files):
                if not fname.endswith(".mp4"):
                    continue

                input_path = os.path.join(root, fname)
                rel_dir = os.path.relpath(root, video_src)

                m = re.match(r"episode_(\d+)\.mp4$", fname)
                if m:
                    # Per-episode file
                    ep_idx = int(m.group(1))
                    if ep_idx not in episodes_to_process:
                        continue
                    mirror_fname = f"episode_{ep_idx + total_original:06d}.mp4"
                    orig_out = os.path.join(video_dst, rel_dir, fname)
                    mirror_out = os.path.join(video_dst, rel_dir, mirror_fname)
                    os.makedirs(os.path.dirname(orig_out), exist_ok=True)
                    count = process_per_episode_video(input_path, orig_out, mirror_out)
                    print(f"  Mirrored {count} frames [episode_{ep_idx}]: {mirror_fname}")
                else:
                    # Chunked file — write one output chunk: orig then mirrored frames
                    out_path = os.path.join(video_dst, rel_dir, fname)
                    if not ordered_ranges:
                        shutil.copy2(input_path, out_path)
                        continue
                    orig_n, mirror_n = process_chunked_video(input_path, out_path, ordered_ranges)
                    print(f"  Chunked video written ({orig_n} orig + {mirror_n} mirrored frames): "
                          f"{os.path.join(rel_dir, fname)}")

    # --- 5. Episodes metadata parquet ---
    write_episodes_metadata(src_dir, out_dir, episodes_to_process, total_original)

    # --- 6. Copy remaining metadata ---
    copy_non_data_files(src_dir, out_dir)

    # --- 7. Update info.json ---
    update_info_json(out_dir, total_episodes=n * 2)

    # --- 8. Upload or report ---
    if args.no_upload:
        print(f"\nDataset saved locally at: {out_dir}")
        username, dataset_name = args.output.split("/", 1)
        url = (
            f"https://huggingface.co/spaces/lerobot/visualize_dataset"
            f"?path=%2F{username}%2F{dataset_name}%2Fepisode_0"
        )
        print(f"Visualizer (after upload): {url}")
    else:
        url = upload_dataset(out_dir, args.output)
        print(f"\nDone! View dataset: {url}")


if __name__ == "__main__":
    main()
