# LeRobot Dataset Augmenter

## 🎯 Purpose
This tool provides an end-to-end automated pipeline to augment and expand **LeRobot v3** robotics datasets. It is specifically designed to address the challenges of imitation learning by diversifying camera observations while maintaining strict synchronization with robot states, actions, and episode indexing.

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
huggingface-cli login
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


## 📦 Core Libraries
* `opencv-python`: High-performance video I/O and frame transformation.
* `pandas` & `pyarrow`: Efficient handling of the large-scale Parquet datasets.
* `huggingface_hub`: Seamless downloading and uploading to the HF ecosystem.
* `argparse`: Provides a thorough command-line interface for tuning augmentation parameters.

---

## 🤖 AI Agent Workflow (Claude Code)
This project was built using Claude Code to rapidly iterate through the LeRobot v3 specification. The development followed a structured "Scaffold-then-Implement" approach based on the following initial prompt:

Initial System Prompt:
"Build a LeRobot v3 dataset augmentation tool. The project structure is:

augment.py: CLI entry point using argparse
transforms.py: video augmentation functions (color_jitter, blur, gaussian_noise, sharpen)
dataset_io.py: download from HF Hub, read/write parquet, upload folder to HF

Key constraints:
Use opencv-python for video frame processing (CPU only, no GPU).

Use huggingface_hub for download/upload.

Use pandas for parquet files.

Parquet files are copied unchanged unless --copies > 1, in which case episode indices are remapped.

Videos are read frame by frame, transformed, written back at same fps.

Must print the HF visualizer link at the end.

Start by scaffolding all three files with stubs, then implement transforms.py first, then dataset_io.py, then augment.py."

How Claude Code was utilized:

* Iterative Scaffolding: The agent first defined the interfaces for each module to ensure augment.py could correctly route CLI arguments to dataset_io.py.

* Logic Implementation: I used Claude to handle the complex math behind the HSV color jitter and unsharp mask sharpening in transforms.py.

* Schema Enforcement: Claude Code was instrumental in writing the update_info_json function, ensuring that the final info.json met the strict requirements for the LeRobot Visualizer.

* Debugging: The agent helped resolve frame-rate synchronization issues when rewriting augmented videos using cv2.VideoWriter.
