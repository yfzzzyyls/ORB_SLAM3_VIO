# EVO Tools Commands for SLAM Evaluation

This document contains the most important `evo` tool commands for evaluating ORB-SLAM3 performance.

## Prerequisites

```bash
# Install evo tools if not already installed
pip install evo --upgrade --no-binary evo

# Activate conda environment
source /home/external/miniconda3/bin/activate
conda activate orbslam
```

## Core Evaluation Commands

### 1. Trajectory Visualization with Alignment

Visualizes and aligns SLAM trajectory with ground truth:

```bash
evo_traj tum results/evo_evaluation_vio/slam_seq135_tum.txt \
         --ref results/evo_evaluation_vio/gt_seq135_tum.txt \
         --plot --plot_mode xyz \
         --save_plot results/trajectories_aligned_vio.pdf --align
```

**Parameters:**
- `--ref`: Reference (ground truth) trajectory
- `--plot`: Show interactive plot
- `--plot_mode xyz`: Show separate X, Y, Z plots
- `--save_plot`: Save plot to PDF
- `--align`: Align trajectories using SE3 Umeyama alignment

### 2. Absolute Trajectory Error (ATE)

Measures global trajectory consistency:

```bash
evo_ape tum results/evo_evaluation/gt_seq135_tum.txt \
        results/evo_evaluation/slam_seq135_tum.txt \
        --align --plot --save_plot ate_over_time_full_slam.pdf
```

**Parameters:**
- First path: Ground truth trajectory
- Second path: SLAM trajectory
- `--align`: Apply SE3 alignment before evaluation
- `--plot`: Show error over time
- `--save_plot`: Save plot to PDF

**Additional useful options:**
- `-v`: Verbose output with statistics
- `--save_results results.zip`: Save detailed results
- `--pose_relation trans_part`: Evaluate only translation error

## Complete Evaluation Pipeline

### Step 1: Convert Trajectories to TUM Format

```bash
# Convert ADT ground truth and SLAM output to TUM format
python scripts/convert_adt_to_tum.py seq135 \
       --slam-file results/f_seq135_trajectory.txt \
       --output-dir results/evo_evaluation
```

### Step 2: Run Full Evaluation Suite

```bash
# ATE - Global consistency
evo_ape tum results/evo_evaluation/gt_seq135_tum.txt \
        results/evo_evaluation/slam_seq135_tum.txt \
        -va --align --plot \
        --save_results results/evo_evaluation/ate_seq135.zip \
        --save_plot results/evo_evaluation/ate_seq135.pdf

# RPE - Frame-to-frame drift
evo_rpe tum results/evo_evaluation/gt_seq135_tum.txt \
        results/evo_evaluation/slam_seq135_tum.txt \
        -va --delta 1 --delta_unit f --plot \
        --save_results results/evo_evaluation/rpe_frame_seq135.zip

# RPE - 1 second intervals (30 frames)
evo_rpe tum results/evo_evaluation/gt_seq135_tum.txt \
        results/evo_evaluation/slam_seq135_tum.txt \
        -va --delta 30 --delta_unit f --plot \
        --save_results results/evo_evaluation/rpe_1s_seq135.zip

# RPE - 1 meter intervals
evo_rpe tum results/evo_evaluation/gt_seq135_tum.txt \
        results/evo_evaluation/slam_seq135_tum.txt \
        -va --delta 1 --delta_unit m --plot \
        --save_results results/evo_evaluation/rpe_1m_seq135.zip
```

## Automated Evaluation Script

For convenience, use the provided evaluation script that runs SLAM and generates both trajectory visualization and error plots:

```bash
# Run VIO (no loop closure) and evaluate with all metrics - DEFAULT
./scripts/evaluate_slam_with_evo.sh seq135 --plot --save-results

# Run full SLAM (with loop closure) and evaluate
./scripts/evaluate_slam_with_evo.sh seq135 --mode slam --plot --save-results

# Skip SLAM execution, only evaluate existing results
./scripts/evaluate_slam_with_evo.sh seq135 --skip-slam --plot

# Evaluate all sequences
./scripts/evaluate_slam_with_evo.sh --all --save-results

# Limit sequence duration (useful for quick tests)
./scripts/evaluate_slam_with_evo.sh seq135 --duration 30 --plot --save-results
```

### Mode Options
- **VIO mode (default)**: Uses `Aria2TUM-VI_VIO.yaml` with `loopClosing: 0`
  - Pure visual-inertial odometry
  - No loop closure corrections
  - Shows real-time drift accumulation
  
- **SLAM mode**: Uses `Aria2TUM-VI.yaml` with loop closure enabled
  - Full SLAM with loop detection
  - Corrects drift when revisiting places
  - Best accuracy for longer sequences

### Generated Outputs
The script now generates all these files in `results/evo_evaluation/`:
- `trajectories_aligned_seq135.pdf` - 3D trajectory visualization (X, Y, Z plots)
- `ate_seq135.pdf` - Absolute trajectory error over time
- `rpe_seq135.pdf` - Relative pose error at 1m intervals
- `rpe_frame_seq135.pdf` - Frame-to-frame relative error
- `rpe_1s_seq135.pdf` - RPE at 1 second intervals
- `rpe_5s_seq135.pdf` - RPE at 5 second intervals

## Understanding the Metrics

### ATE (Absolute Trajectory Error)
- **What it measures**: Global consistency of the entire trajectory
- **Units**: Meters
- **Interpretation**: 
  - < 0.1m: Excellent
  - 0.1-0.5m: Good
  - 0.5-1.0m: Acceptable
  - > 1.0m: Poor

### RPE (Relative Pose Error)
- **What it measures**: Local drift between poses
- **Units**: Meters (translation) or degrees (rotation)
- **Delta units**:
  - `f`: Frames (e.g., frame-to-frame)
  - `m`: Meters (e.g., every meter traveled)
  - `s`: Seconds (not directly supported, use frames)
- **Interpretation**: RPE shows drift rate, NOT cumulative error
  - 0.7° per frame = local rotation error between consecutive frames
  - Does NOT mean 21°/second cumulative drift

## Other Useful Commands

### Compare Multiple Results

```bash
# Generate comparison table from multiple runs
evo_res results/evo_evaluation/ate_*.zip -p \
        --save_table results/comparison_table.csv
```

### Filter Trajectory by Time

```bash
# Evaluate only first 60 seconds
evo_ape tum gt.txt slam.txt --t_start 0 --t_end 60
```

### Export Aligned Trajectories

```bash
# Save aligned trajectories for further analysis
evo_traj tum slam.txt --ref gt.txt --align \
         --save_as_tum slam_aligned.txt
```

## Tips and Best Practices

1. **Always align trajectories** when comparing monocular SLAM (scale ambiguity)
2. **Check timestamp synchronization** before evaluation
3. **Use consistent delta units** for RPE comparison
4. **Save results as .zip** for later analysis and comparison
5. **Plot errors over time** to identify when drift occurs

## Troubleshooting

### "No common timestamps found"
- Check timestamp formats (should be seconds with decimal)
- Verify trajectories overlap in time
- Use `--t_offset` to adjust timestamp offset

### "Scale difference too large"
- Monocular SLAM has scale ambiguity
- Always use `--align` for monocular systems
- Check if IMU initialization succeeded

### "Too few poses"
- Ensure SLAM completed successfully
- Check if tracking was lost frequently
- Verify data paths are correct