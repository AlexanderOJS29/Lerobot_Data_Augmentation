# LeRobot Augmentation Tool

## Purpose
Augment LeRobot v3 HuggingFace datasets with visual transforms.

## Key libraries
- opencv-python (cv2): video I/O and frame transforms
- huggingface_hub: dataset download/upload
- pandas: parquet file handling
- datasets: HF dataset loading
- argparse: CLI

## LeRobot v3 dataset structure
- data/train/episode_XXXXXX.parquet — per-episode robot states/actions
- videos/chunk-000/<camera_name>/episode_XXXXXX.mp4 — video per camera
- meta/info.json — dataset schema, fps, total_episodes, features
- meta/tasks.jsonl — language instructions
- meta/episodes/ — per-episode metadata files

## Critical rules
- Parquet columns: observation.state, action, timestamp, episode_index, 
  frame_index, task_index, next.reward, next.done
- Episode indices must be contiguous starting from 0
- Video fps must match info.json fps field
- Each camera gets its own video subfolder

## Commands
- Run: python augment.py --source <hf_path> --output <username/repo>
- Test: python augment.py --source lerobot/aloha_static_cups_open --output <username>/test --episodes 2