# LeRobot Dataset Augmenter

## 🎯 Purpose
This project provides an end-to-end automated pipeline to augment and expand **LeRobot v3** robotics datasets. It includes two tools:

- **`augment.py`** — applies visual transforms (color jitter, blur, noise, sharpen) to camera streams
- **`mirror_dataset.py`** — generates horizontally-mirrored trajectory copies, doubling the dataset with negated action dimensions

Both tools are specifically designed to address the challenges of imitation learning by diversifying demonstrations while maintaining strict synchronization with robot states, actions, and episode indexing.

## 🏗️ How it Handles Data
The tool performs a deep transformation of the LeRobot v3 data structure to ensure the output is ready for training and visualization:

* **Video Transformation:** Camera streams (stored in `videos/`) are processed frame-by-frame using OpenCV. The tool is sophisticated enough to handle both "concatenated chunk" videos (where multiple episodes are in one file) and "per-episode" files.
* **Parquet Synchronization:** All state and action data stored in `.parquet` files are read into memory, filtered if necessary, and rewritten to match the augmented video frames.
* **Episode Remapping:** If creating multiple copies of a dataset, the tool automatically calculates and applies offsets to the `episode_index` and `frame_index` columns to ensure the new dataset remains contiguous and valid for the LeRobot API.
* **Metadata Updates:** The `meta/info.json` file is patched with updated `total_episodes` and `total_frames` counts, and dataset "splits" (train/test) are recalculated based on the new data volume.

---

## 🎨 Augmentation Suite
The tool provides several CPU-optimized computer vision transforms via OpenCV:

| Transform | Flag | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Color Jitter** | `color_jitter` | `--brightness`, `--contrast`, `--saturation`, `--hue` | Randomly shifts pixel intensity and color in HSV space to simulate various lighting conditions. |
| **Gaussian Noise** | `gaussian_noise` | `--noise-std` | Adds random grain to the image to simulate sensor noise or transmission interference. |
| **Blur** | `blur` | `--kernel-size` | Applies Gaussian blurring to simulate motion blur or camera focus issues. |
| **Sharpen** | `sharpen` | `--sharpen-strength` | Uses an unsharp mask to enhance edges, simulating high-definition sensors. |

---

## 🛠️ Installation & Setup

### 1. Conda Environment
It is recommended to use **Python 3.12.9** for this environment to ensure compatibility with the latest data processing libraries.

```bash
# Create the environment with specific Python version
conda create -n lerobot-aug python=3.12.9 -y
conda activate lerobot-aug

# Install required dependencies
pip install opencv-python pandas huggingface_hub pyarrow
```

### 2. Hugging Face Authentication
The tool interacts directly with the Hugging Face Hub to download source datasets and upload augmented results:
```bash
hf auth login
```

---

## 🚀 Execution Guide

### Choosing Augmentations & Episode Counts
To run the tool, you must specify the **source** (HF Hub ID), the **output** (your target repository), the **transforms**, and the number of **episodes** to process.

**Example Command:**
Process the first **10 episodes** of a dataset using **Color Jitter** and **Gaussian Noise**:
```bash
python augment.py \
    --source lerobot/aloha_static_cups_open \
    --output <your-username>/augmented-aloha-10ep \
    --episodes 10 \
    --transforms color_jitter gaussian_noise \
    --brightness 0.3 \
    --noise-std 15.0
```
Example visualization: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FASet97%2Faugmented-aloha-episodes-10-color-jitter-gaussian-noise%2Fepisode_0

### Dataset Multiplication (Expansion)
You can create multiple augmented copies of each episode to expand your training data. The command below creates **3 copies** for every 1 original episode, automatically remapping indices:
```bash
python augment.py \
    --source lerobot/pusht \
    --output <your-username>/expanded-pusht \
    --copies 3 \
    --transforms color_jitter
```

### 📂 Local Processing (No Upload)

If you do not want to upload the augmented dataset to the Hugging Face Hub, use the --no-upload flag. This is highly recommended for initial testing or when working with sensitive data.

When this flag is used, the tool will save all processed files (Parquets, Videos, and Metadata) to a temporary directory or a directory of your choice.

Example: Local-only run

```bash
python augment.py \
    --source lerobot/pusht \
    --output local_folder_name \
    --no-upload \
    --episodes 2 \
    --transforms color_jitter
```    
Note: You can also use --work-dir <path> to specify exactly where on your machine the augmented files should be saved. If not specified, the tool creates a unique temporary directory for the run.

---

## 🪞 Trajectory Mirroring (`mirror_dataset.py`)

The mirroring tool doubles a dataset by producing a horizontally-flipped copy of every episode. Each mirrored copy has its action dimensions negated and is assigned a new episode index (`original_index + total_original_episodes`), so the output is a single valid dataset containing both the originals and their mirrors.

### What it does per episode
* **Video:** every frame is flipped horizontally with `cv2.flip(frame, 1)` across all cameras
* **Actions:** the action dimensions listed in `--mirror-dims` are negated (e.g. left/right and rotation axes)
* **Episode indices:** mirrored episodes are offset by `total_episodes` from the source dataset so indices remain contiguous
* **Metadata:** `meta/episodes/chunk-000/file-000.parquet` is extended with remapped frame ranges and timestamps; `meta/info.json` is updated with the doubled episode count

### Example Commands

**Basic run — 10 episodes, upload to HF:**
```bash
python mirror_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output <your-username>/mirrored-aloha \
    --episodes 10
```

**Custom mirror dimensions (default is `0,2`):**
```bash
python mirror_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output <your-username>/mirrored-aloha \
    --mirror-dims 0,1,2
```

**Local test run without uploading:**
```bash
python mirror_dataset.py \
    --source lerobot/aloha_static_cups_open \
    --output <your-username>/mirrored-aloha \
    --episodes 2 \
    --no-upload
```

### CLI Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | *(required)* | HF repo ID of the source dataset |
| `--output` | *(required)* | HF repo ID for the output dataset |
| `--mirror-dims` | `0,2` | Comma-separated action dimension indices to negate |
| `--episodes` | all | Only process the first N episodes (useful for testing) |
| `--no-upload` | off | Skip HF upload, save output locally only |
| `--work-dir` | system temp | Custom working directory for intermediate files |

---

## 📦 Core Libraries
* `opencv-python`: High-performance video I/O and frame transformation.
* `pandas` & `pyarrow`: Efficient handling of the large-scale Parquet datasets.
* `huggingface_hub`: Seamless downloading and uploading to the HF ecosystem.
* `argparse`: Provides a thorough command-line interface for tuning augmentation parameters.

---

## 🤖 AI Agent Workflow (Claude Code)

This project was built iteratively with **Claude Code** (Sonnet 4.6) acting as a hands-on engineering collaborator — reading code, writing files, running tests, diagnosing failures, and patching bugs in the same session.

---

### Phase 1 — Augmentation Pipeline (`augment.py`, `transforms.py`, `dataset_io.py`)

The initial build followed a structured **Scaffold-then-Implement** approach:

**Initial prompt:**
> "Build a LeRobot v3 dataset augmentation tool. The project structure is:
> `augment.py` (CLI), `transforms.py` (video transforms), `dataset_io.py` (HF Hub I/O).
> Use opencv-python for video frame processing (CPU only). Use huggingface_hub for download/upload. Use pandas for parquet files. Parquet files are copied unchanged unless `--copies > 1`, in which case episode indices are remapped. Videos are read frame by frame, transformed, written back at the same fps. Must print the HF visualizer link at the end."

**What Claude Code did:**

* **Module scaffolding:** Defined all three files with matching interfaces before implementing any logic, so the CLI argument routing was correct from the start.
* **Transform math:** Implemented the HSV color jitter (per-channel random shifts in HSV space) and the unsharp mask for the `sharpen` transform.
* **Chunked video handling:** Identified that LeRobot v3 concatenates all episodes into chunk files (`file-000.mp4`) rather than per-episode files, and built the frame-seeking logic in `transform_video` to process only the relevant frame range using `cap.set(cv2.CAP_PROP_POS_FRAMES, start)`.
* **Episode remapping:** Wrote `remap_episodes()` to apply index offsets when `--copies > 1`, ensuring the output parquet stays contiguous.
* **Metadata patching:** Implemented `update_info_json()` to patch `total_episodes`, `total_frames`, and `splits` so the HF visualizer accepts the output dataset.
* **Codec fallback:** Added an `avc1 → mp4v` fallback in `cv2.VideoWriter` after discovering that `avc1` is not always available depending on the OpenCV build.

---

### Phase 2 — Trajectory Mirroring (`mirror_dataset.py`)

After the augmentation pipeline was stable, a second tool was requested: a mirroring tool that doubles the dataset by producing a horizontally-flipped copy of every trajectory.

**Design constraint:** reuse all I/O from `dataset_io.py` — no duplicated download, upload, or metadata logic.

**What Claude Code did:**

* **Architecture:** Designed `mirror_dataset.py` to import `download_dataset`, `copy_non_data_files`, `upload_dataset`, and `update_info_json` directly from `dataset_io.py`, keeping the file focused purely on mirroring logic.
* **Action negation:** Implemented `mirror_parquet()` to negate configurable action dimensions (`--mirror-dims`) per row, handling variable-length action vectors safely.
* **Chunked video mirroring:** Built `process_chunked_video()` to make two passes over the source chunk — one writing original frames and one writing flipped frames — into a single valid output `file-000.mp4`, avoiding the broken `_mirror` suffix naming that appeared in the first run.
* **Live test and bug fix (round 1):** On the first local test run, Claude spotted that the chunked video was processing all 50 source episodes (20,000 frames) instead of just the 2 requested. Fixed by computing per-episode frame ranges from the episodes metadata parquet and passing them to a seek-based reader.
* **Live test and bug fix (round 2):** After running 10 real episodes, the HF visualizer returned *"Episode 0 not found in metadata"*. Claude diagnosed that `meta/episodes/chunk-000/file-000.parquet` was never written — `copy_non_data_files` skips parquet files, and the data walk only covers `data/`. Fixed by implementing `write_episodes_metadata()`, which reads the source episodes parquet, filters to the processed episodes, remaps `dataset_from_index`/`dataset_to_index` to be contiguous from 0, creates mirrored rows with the correct frame offsets, and writes the combined file to the output.

---

### Development Principles Applied

Throughout both phases Claude Code followed these rules, which kept the codebase clean:

* **No duplication** — `mirror_dataset.py` shares all I/O with `dataset_io.py` rather than reimplementing it.
* **No speculative abstractions** — helpers were only added when a concrete need appeared (e.g. `write_episodes_metadata` was added after the visualizer error, not preemptively).
* **Diagnose before switching tactics** — both video bugs were diagnosed by reading error output and inspecting the actual parquet/video data before changing the approach.
* **Test first, then ship** — each tool was run locally with `--no-upload --episodes 2` and output verified before the real upload command was handed back.
