"""CLI entry point for LeRobot v3 dataset augmentation."""

import argparse
import json
import os
import tempfile

from transforms import TRANSFORMS
from dataset_io import (
    download_dataset,
    transform_videos,
    copy_and_remap_parquet,
    copy_non_data_files,
    upload_dataset,
    update_info_json,
    get_episode_frame_ranges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot v3 dataset with video transforms."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="HF Hub dataset repo id to augment (e.g. lerobot/pusht)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="HF Hub repo id for the augmented dataset (e.g. username/repo)",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=list(TRANSFORMS.keys()),
        default=["color_jitter"],
        help="Transforms to apply to video frames (default: color_jitter)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Number of copies (1 = just transform in-place, >1 = duplicate episodes with remapped indices)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Only process the first N episodes (useful for testing)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for downloads/outputs (default: temp dir)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HF Hub (useful for local testing)",
    )

    # Transform-specific params
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--saturation", type=float, default=0.2)
    parser.add_argument("--hue", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=10.0)
    parser.add_argument("--sharpen-strength", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="lerobot_aug_")
    src_dir = os.path.join(work_dir, "source")
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all transform-specific params into one dict.
    # apply_transforms() will route each kwarg to the right function.
    transform_kwargs = {
        "brightness": args.brightness,
        "contrast": args.contrast,
        "saturation": args.saturation,
        "hue": args.hue,
        "kernel_size": args.kernel_size,
        "std": args.noise_std,
        "strength": args.sharpen_strength,
    }

    # Step 1: Download
    print(f"=== Downloading {args.source} ===")
    src_dir = download_dataset(args.source, src_dir)

    # Step 2: If --episodes is set, restrict processing to the first N episodes.
    # Frame ranges are needed because LeRobot v3 videos are concatenated — we
    # need to know which frames in the video belong to which episodes.
    episode_indices = None
    episode_frame_ranges = None
    if args.episodes is not None:
        print(f"=== Filtering to first {args.episodes} episodes ===")
        episode_indices = list(range(args.episodes))
        episode_frame_ranges = get_episode_frame_ranges(src_dir, episode_indices)
        print(f"  Found frame ranges for {len(episode_frame_ranges)} episodes")

    # Step 3: Copy metadata files
    print("=== Copying metadata ===")
    copy_non_data_files(src_dir, out_dir)

    # Step 4: Copy/remap parquet files (filtered if needed)
    print("=== Processing parquet files ===")
    copy_and_remap_parquet(src_dir, out_dir, args.copies, episode_indices=episode_indices)

    # Step 5: Transform videos
    print(f"=== Transforming videos with {args.transforms} ===")

    # LeRobot v3 layout: videos/<camera_name>/chunk-NNN/file-NNN.mp4
    src_video_dir = os.path.join(src_dir, "videos")
    out_video_dir = os.path.join(out_dir, "videos")

    if os.path.isdir(src_video_dir):
        if args.copies <= 1:
            transform_videos(
                src_video_dir, out_video_dir, args.transforms,
                episode_indices=episode_indices,
                episode_frame_ranges=episode_frame_ranges, **transform_kwargs,
            )
        else:
            for copy_idx in range(args.copies):
                print(f"  --- Copy {copy_idx + 1}/{args.copies} ---")
                copy_video_out = os.path.join(work_dir, f"videos_copy_{copy_idx}")
                transform_videos(
                    src_video_dir, copy_video_out, args.transforms,
                    episode_indices=episode_indices,
                    episode_frame_ranges=episode_frame_ranges, **transform_kwargs,
                )

                # Move transformed videos into the final output dir.
                # For copies > 0 with per-episode filenames, offset episode numbers
                # so they don't collide (e.g. copy 1's ep 0 becomes ep N).
                for root, _dirs, files in os.walk(copy_video_out):
                    for fname in sorted(files):
                        if not fname.endswith(".mp4"):
                            continue
                        src_path = os.path.join(root, fname)
                        rel_path = os.path.relpath(src_path, copy_video_out)

                        if copy_idx == 0:
                            dst_path = os.path.join(out_video_dir, rel_path)
                        else:
                            # Rename: episode_000000.mp4 -> episode_NNNNNN.mp4
                            parts = rel_path.split(os.sep)
                            base = parts[-1]
                            if base.startswith("episode_"):
                                ep_num_str = base.replace("episode_", "").replace(".mp4", "")
                                try:
                                    original_ep = int(ep_num_str)
                                    num_orig = len(episode_indices) if episode_indices else len([
                                        f for f in os.listdir(os.path.join(src_video_dir, *parts[:-1]))
                                        if f.startswith("episode_") and f.endswith(".mp4")
                                    ])
                                    new_ep = original_ep + copy_idx * num_orig
                                    parts[-1] = f"episode_{new_ep:06d}.mp4"
                                except ValueError:
                                    pass
                            dst_path = os.path.join(out_video_dir, *parts)

                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        os.rename(src_path, dst_path)
    else:
        print("  No videos/ directory found, skipping video transforms.")

    # Step 6: Sync meta/info.json with the actual output contents.
    # total_episodes, total_frames, and splits must all reflect what we produced.
    print("=== Updating meta/info.json ===")
    num_episodes = len(episode_indices) if episode_indices else _count_original_episodes(src_dir)
    total_ep = num_episodes * max(args.copies, 1)
    # Derive total_frames from the output parquet rather than computing it
    data_parquet = os.path.join(out_dir, "data", "chunk-000", "file-000.parquet")
    if os.path.exists(data_parquet):
        import pandas as pd
        total_frames = len(pd.read_parquet(data_parquet, columns=["episode_index"]))
    else:
        total_frames = None
    update_info_json(out_dir, total_episodes=total_ep, total_frames=total_frames)

    # Step 7: Upload
    if args.no_upload:
        print(f"\n=== Output saved to {out_dir} (upload skipped) ===")
    else:
        print(f"=== Uploading to {args.output} ===")
        url = upload_dataset(out_dir, args.output)
        print(f"\n=== Done! ===")
        print(f"Visualize your dataset: {url}")


def _count_original_episodes(src_dir: str) -> int:
    """Read total_episodes from source info.json, or count from parquet."""
    info_path = os.path.join(src_dir, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        return info.get("total_episodes", 0)
    return 0


if __name__ == "__main__":
    main()
