"""Download/upload LeRobot v3 datasets via HF Hub, read/write parquet with pandas."""

import json
import os
import re
import shutil

import cv2
import pandas as pd
from huggingface_hub import snapshot_download, HfApi
from transforms import apply_transforms


def download_dataset(repo_id: str, local_dir: str) -> str:
    """Download a dataset from HF Hub. Returns the local path."""
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print(f"Downloaded {repo_id} to {path}")
    return path


def read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a parquet file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def remap_episodes(df: pd.DataFrame, copy_index: int, num_original_episodes: int) -> pd.DataFrame:
    """Remap episode indices for a duplicated copy of the dataset.

    copy_index 0 = original (no change), 1 = first augmented copy, etc.
    Each copy's episode indices are offset by copy_index * num_original_episodes.
    """
    remapped = df.copy()
    if "episode_index" in remapped.columns:
        remapped["episode_index"] = remapped["episode_index"] + copy_index * num_original_episodes
    return remapped


def filter_episodes(df: pd.DataFrame, episode_indices: list[int]) -> pd.DataFrame:
    """Filter a DataFrame to only include rows for the given episode indices."""
    if "episode_index" not in df.columns:
        return df
    return df[df["episode_index"].isin(episode_indices)].reset_index(drop=True)


def get_episode_frame_ranges(src_dir: str, episode_indices: list[int]) -> dict[int, tuple[int, int]]:
    """Get (from_frame, to_frame) for each requested episode.

    Returns {episode_index: (first_global_frame, last_global_frame + 1)}.
    Used to seek into concatenated video files that contain all episodes.
    """
    # Preferred: episodes metadata has precomputed dataset_from/to_index
    ep_parquet = os.path.join(src_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    if os.path.exists(ep_parquet):
        ep_df = pd.read_parquet(ep_parquet)
        if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
            ranges = {}
            for _, row in ep_df.iterrows():
                ep_idx = int(row["episode_index"])
                if ep_idx in episode_indices:
                    ranges[ep_idx] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
            return ranges

    # Fallback: derive ranges from row positions in the data parquet
    data_parquet = os.path.join(src_dir, "data", "chunk-000", "file-000.parquet")
    if os.path.exists(data_parquet):
        df = pd.read_parquet(data_parquet, columns=["episode_index", "frame_index"])
        ranges = {}
        for ep_idx in episode_indices:
            ep_data = df[df["episode_index"] == ep_idx]
            if len(ep_data) == 0:
                continue
            # Global frame position = cumulative count up to this episode
            first_row = ep_data.index[0]
            ranges[ep_idx] = (first_row, first_row + len(ep_data))
        return ranges

    return {}


def transform_video(input_path: str, output_path: str, transform_names: list[str],
                    frame_range: tuple[int, int] | None = None, **kwargs) -> None:
    """Read a single video frame by frame, apply transforms, write at same fps.

    If frame_range is (start, end), only process frames in [start, end).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Prefer H.264 (avc1) to match original LeRobot videos and HF visualizer expectations
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("avc1 unavailable")
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Seek to start frame if only processing a subset (e.g. --episodes filter)
    if frame_range is not None:
        start_frame, end_frame = frame_range
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        augmented = apply_transforms(frame, transform_names, **kwargs)
        writer.write(augmented)
        frame_count += 1
        current_frame += 1

    cap.release()
    writer.release()
    print(f"  Transformed {frame_count} frames: {os.path.basename(output_path)}")


def _episode_index_from_filename(fname: str) -> int | None:
    """Extract episode index from a filename like episode_000003.mp4 or .parquet."""
    m = re.match(r"episode_(\d+)\.", fname)
    return int(m.group(1)) if m else None


def transform_videos(video_dir: str, output_dir: str, transform_names: list[str],
                     episode_indices: list[int] | None = None,
                     episode_frame_ranges: dict[int, tuple[int, int]] | None = None,
                     **kwargs) -> None:
    """Walk videos/<camera>/chunk-NNN/ and transform each .mp4, preserving directory layout.

    LeRobot v3 concatenates all episodes into chunked files (e.g. file-000.mp4).
    When --episodes is used, episode_frame_ranges maps episode indices to global
    frame offsets so we can seek into the concatenated video and only process
    the relevant portion.
    """
    for root, _dirs, files in os.walk(video_dir):
        for fname in sorted(files):
            if not fname.endswith(".mp4"):
                continue

            input_path = os.path.join(root, fname)
            rel_path = os.path.relpath(input_path, video_dir)

            # Some datasets use per-episode files (episode_XXXXXX.mp4) — filter by name
            ep_idx = _episode_index_from_filename(fname)
            if ep_idx is not None:
                if episode_indices is not None and ep_idx not in episode_indices:
                    continue
                output_path = os.path.join(output_dir, rel_path)
                transform_video(input_path, output_path, transform_names, **kwargs)
                continue

            # Chunked files (file-000.mp4): all episodes concatenated in one video
            if episode_indices is not None and episode_frame_ranges:
                # Merge episode ranges into one contiguous span (assumes episodes are adjacent)
                ranges = sorted(episode_frame_ranges.values())
                if not ranges:
                    continue
                start = ranges[0][0]
                end = ranges[-1][1]
                output_path = os.path.join(output_dir, rel_path)
                transform_video(input_path, output_path, transform_names,
                                frame_range=(start, end), **kwargs)
            else:
                output_path = os.path.join(output_dir, rel_path)
                transform_video(input_path, output_path, transform_names, **kwargs)


def copy_and_remap_parquet(src_dir: str, dst_dir: str, copies: int,
                           episode_indices: list[int] | None = None) -> None:
    """Copy parquet files. If copies > 1, concatenate with remapped episode indices.

    Handles LeRobot v3 structure where data/ has chunked parquet files containing all episodes.
    If episode_indices is set, filters to only those episodes.
    """
    parquet_files = []
    for root, _dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith(".parquet"):
                continue
            # Filter per-episode parquet files by name if applicable
            if episode_indices is not None:
                ep_idx = _episode_index_from_filename(fname)
                if ep_idx is not None and ep_idx not in episode_indices:
                    continue
            parquet_files.append(os.path.join(root, fname))

    for pq_path in parquet_files:
        rel_path = os.path.relpath(pq_path, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)

        if os.path.join("meta", "episodes") in pq_path:
            # Copy episode metadata as-is, don't remap
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(pq_path, dst_path)
            continue

        df = read_parquet(pq_path)

        # Filter by episode for chunked files (not per-episode named)
        if episode_indices is not None and _episode_index_from_filename(os.path.basename(pq_path)) is None:
            df = filter_episodes(df, episode_indices)

        if copies <= 1:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            write_parquet(df, dst_path)
        else:
            num_episodes = df["episode_index"].nunique() if "episode_index" in df.columns else 0
            frames = []
            for i in range(copies):
                frames.append(remap_episodes(df, i, num_episodes))
            combined = pd.concat(frames, ignore_index=True)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            write_parquet(combined, dst_path)
            print(f"  Parquet remapped ({copies} copies): {rel_path}")


def copy_non_data_files(src_dir: str, dst_dir: str) -> None:
    """Copy metadata files (json, jsonl, yaml, md, etc.) that aren't parquet or video."""
    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            if fname.endswith((".parquet", ".mp4")):
                continue
            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, src_dir)
            # Skip hidden files/dirs (.cache, .gitattributes, etc.)
            if any(part.startswith(".") for part in rel_path.split(os.sep)):
                continue
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


def update_info_json(output_dir: str, total_episodes: int, total_frames: int | None = None) -> None:
    """Update meta/info.json with new total_episodes, total_frames, and splits."""
    info_path = os.path.join(output_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        print("  Warning: meta/info.json not found, skipping update")
        return

    with open(info_path) as f:
        info = json.load(f)

    old_ep = info.get("total_episodes", "?")
    info["total_episodes"] = total_episodes

    if total_frames is not None:
        info["total_frames"] = total_frames

    # Update splits to reflect new episode range
    if "splits" in info:
        info["splits"] = {"train": f"0:{total_episodes}"}

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Updated meta/info.json: total_episodes {old_ep} -> {total_episodes}"
          f", total_frames -> {total_frames}, splits -> 0:{total_episodes}")


def upload_dataset(local_dir: str, repo_id: str) -> str:
    """Upload augmented dataset folder to HF Hub. Returns the visualizer URL."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    username, dataset_name = repo_id.split("/", 1)
    url = (
        f"https://huggingface.co/spaces/lerobot/visualize_dataset"
        f"?path=%2F{username}%2F{dataset_name}%2Fepisode_0"
    )
    return url
