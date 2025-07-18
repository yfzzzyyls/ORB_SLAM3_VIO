# Scripts Directory

This directory contains utility scripts for running ORB-SLAM3 on Aria Digital Twin (ADT) data and analyzing the results.

## Important Note on Analysis Scripts

The analysis scripts require SLAM to have been run first:

- **`analyze_gaze_slam_hit_rate.py`** - Analyzes SLAM features on gazed objects
- **`analyze_slam_depth_standard_metrics.py`** - Compares SLAM depth with ground truth

These scripts use:
- **ADT data** from `/mnt/ssd_ext/incSeg-data/adt/` (VRS files, gaze, segmentation)
- **SLAM tracking data** from `/home/external/ORB_SLAM3_VIO/results/tracking_data_*`

**Current status:**
- Default tracking directories (`tracking_data_*_trajectory`) only exist for test sequences (seq141, seq142)
- Training sequences (seq131-138, 140) have tracking data for specific feature caps:
  - `--data-dir cap1000_trajectory` for 1000 feature cap results
  - `--data-dir cap500_trajectory` for 500 feature cap results

To generate default tracking data, run SLAM with `ORBextractor.nFeatures: 2000` in the config file.

## Core Pipeline Scripts

### `run_orbslam3.sh`
Runs ORB-SLAM3 on a single ADT sequence.
```bash
# Usage
export SAVE_TRACKING=1
./run_orbslam3.sh --all

# Example
./run_orbslam3.sh Apartment_release_clean_seq131_M1292
```

### `processed_adt_to_tumvi.py`
Converts processed ADT data to TUM-VI format for ORB-SLAM3.
```bash
# Basic usage
python processed_adt_to_tumvi.py <input_dir> <output_dir>

# With duration limit
python processed_adt_to_tumvi.py processed_adt/seq131 output_tumvi/seq131 --duration 30

# Use left camera instead of right (not recommended)
python processed_adt_to_tumvi.py processed_adt/seq131 output_tumvi/seq131 --use-left-camera
```

### `extract_adt.py`
Extracts data from ADT VRS files to processed format.
```bash
python extract_adt.py
```

## Analysis Scripts

### `analyze_gaze_slam_hit_rate.py`
Analyzes how often gazed objects have at least one SLAM feature point.
```bash
# Analyze default tracking data (2000 features)
# NOTE: Default directories only exist if you've run SLAM without feature caps
# Currently only seq141, seq142 have default tracking data
python analyze_gaze_slam_hit_rate.py

# Analyze specific feature cap (recommended if default doesn't exist)
python analyze_gaze_slam_hit_rate.py --data-dir cap1000_trajectory
python analyze_gaze_slam_hit_rate.py --data-dir cap500_trajectory

# To generate default tracking data for all sequences:
# 1. Set ORBextractor.nFeatures: 2000 in Aria2TUM-VI.yaml
# 2. Run: ./run_all_sequences.sh (or run_orbslam3.sh for each sequence)
```

### `analyze_slam_depth_standard_metrics.py`
Calculates standard depth evaluation metrics (Abs Rel, RMSE, threshold accuracies) for SLAM vs ground truth.
```bash
# Analyze default tracking data (2000 features)
# NOTE: Default directories only exist if you've run SLAM without feature caps
python analyze_slam_depth_standard_metrics.py

# Analyze specific feature cap (recommended if default doesn't exist)
python analyze_slam_depth_standard_metrics.py --data-dir cap1000_trajectory
python analyze_slam_depth_standard_metrics.py --data-dir cap500_trajectory
```

### `analyze_slam_pose_accuracy.py`
Compares SLAM camera poses with ground truth trajectories from ADT.
```bash
# Analyze default trajectory files (2000 features)
python analyze_slam_pose_accuracy.py

# Analyze specific feature cap
python analyze_slam_pose_accuracy.py --data-dir cap1000_trajectory
python analyze_slam_pose_accuracy.py --data-dir cap500_trajectory
```
Metrics include:
- **ATE** (Absolute Trajectory Error): Global trajectory consistency
- **RPE** (Relative Pose Error): Frame-to-frame drift in translation and rotation

### `visualize_overlapping.py`
Visualizes gaze points, SLAM features, and segmentation masks overlaid on images.
```bash
# Visualize specific frames
python visualize_overlapping.py --seq 135 --frames 500 1000 1500

# Continuous visualization
python visualize_overlapping.py --seq 135 --start-frame 500

# Save output video
python visualize_overlapping.py --seq 135 --frames 500 1000 --save-video output.mp4
```

## Batch Processing Scripts

### `run_all_sequences_with_caps.sh`
Runs ORB-SLAM3 on all sequences with different feature caps (1000, 500).
```bash
./run_all_sequences_with_caps.sh
```

## Analysis Results

### `feature_cap_analysis_results.md`
Summary report of feature reduction experiments showing:
- Hit rates for different feature caps
- Depth accuracy metrics
- Recommendations for production use

## Environment Setup

Before running any scripts, ensure you have:
1. Activated the conda environment: `conda activate orbslam`
2. Set up ORB-SLAM3 environment: `source ../setup_env.sh`
3. For SLAM tracking data save: `export SAVE_TRACKING=1`

## Directory Structure

Scripts expect the following data structure:
- Input VRS files: `/mnt/ssd_ext/incSeg-data/adt/{train,test}/*/`
- Processed ADT data: `processed_adt/`
- TUM-VI output: `output_tumvi/`
- Tracking results: `/home/external/ORB_SLAM3_VIO/results/tracking_data_*`

## Feature Cap Experiments

To analyze impact of feature reduction:
1. Run SLAM with different caps using `run_all_sequences_with_caps.sh`
2. Analyze hit rates with `analyze_gaze_slam_hit_rate.py --data-dir cap{N}_trajectory`
3. Analyze depth accuracy with `analyze_slam_depth_standard_metrics.py --data-dir cap{N}_trajectory`
4. See `feature_cap_analysis_results.md` for comprehensive results

### Key Findings
- Default (2000): ~89.7% hit rate, best accuracy
- Cap 1000: 86.0% hit rate, 76.2% within 1.25x threshold
- Cap 500: 80.5% hit rate, 59.3% within 1.25x threshold
- Cap 200: Tracking failure

## Notes

- Scripts analyze frames 500-3000 to ensure SLAM has stabilized
- All scripts support ADT train sequences (seq131-138, 140)
- Default tracking directories only exist if you've run SLAM without feature caps
- For feature cap analysis, use the `--data-dir` argument