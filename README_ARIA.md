# Aria Everyday Activities Dataset for ORB-SLAM3

This guide explains how to process and run Aria Everyday Activities (AEA) dataset with ORB-SLAM3.

## Successfully Running ORB-SLAM3 on Aria Data (2025-07-01)

### Key Achievements
1. **Data Conversion**: Successfully converts Aria VRS to TUM-VI format
   - 480×640 images with 90° rotation applied
   - IMU at 100Hz (downsampled from 1000Hz with interpolation)
   - Proper nanosecond timestamps

2. **Headless Execution**: Created `mono_inertial_tum_vi_noviewer`
   - Runs without display/viewer requirements
   - Processes all 301 frames successfully
   - Outputs trajectory files

3. **Results**:
   - Map created with 313 points and 42 keyframes
   - Median tracking time: 13.9ms
   - Full trajectory saved (275 poses)
   - IMU preintegration working (confirmed by "VIBA" messages)

### Complete Step-by-Step Instructions

#### Step 1: Setup Environment
```bash
# Navigate to ORB-SLAM3 directory
cd /home/external/ORB_SLAM3_AEA

# Activate Python environment for Aria tools
source ~/venv/py39/bin/activate

# Setup ORB-SLAM3 environment
source setup_env.sh
```

#### Step 2: Build ORB-SLAM3 (if not already built)
```bash
# Option 1: Use the build script (recommended)
./build.sh

# Option 2: Build manually
mkdir -p build
cd build
cmake ..
make -j4
cd ..
```

#### Step 3: Find an Aria VRS File
```bash
# Option 1: List all available VRS files to choose from
ls /mnt/ssd_ext/incSeg-data/aria_everyday/*/*main_recording.vrs

# Option 2: Find a specific sequence (e.g., location 1, recording 1)
VRS_FILE=$(find /mnt/ssd_ext/incSeg-data/aria_everyday -name "*main_recording.vrs" | grep "loc1_script1_seq1_rec1" | head -1)
echo "Found: $VRS_FILE"

# Option 3: Use a specific known path
VRS_FILE="/mnt/ssd_ext/incSeg-data/aria_everyday/loc1_script1_seq1_rec1/AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_main_recording.vrs"

# Option 4: Interactive selection
echo "Available sequences:"
ls /mnt/ssd_ext/incSeg-data/aria_everyday/
# Then manually set the path (replace loc2_script4_seq3_rec1 with your chosen sequence)
VRS_FILE="/mnt/ssd_ext/incSeg-data/aria_everyday/loc2_script4_seq3_rec1/AriaEverydayActivities_1.0.0_loc2_script4_seq3_rec1_main_recording.vrs"
```

#### Step 4: Convert VRS to TUM-VI Format
```bash
# Quick test with 30 seconds of data
python aria_to_tumvi.py "$VRS_FILE" aria_tumvi_test --duration 30 --imu-freq 100

# Or convert full sequence (may take several minutes)
python aria_to_tumvi.py "$VRS_FILE" aria_tumvi_full --imu-freq 100
```

#### Step 5: Run ORB-SLAM3
```bash
# Using the convenience script (recommended)
./run_orbslam3_aria_tumvi_headless.sh aria_tumvi_test my_trajectory

# Or run directly
./Examples/Monocular-Inertial/mono_inertial_tum_vi_noviewer \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
    aria_tumvi_test/mav0/cam0/data \
    aria_tumvi_test/mav0/timestamps.txt \
    aria_tumvi_test/mav0/imu0/data.csv \
    my_trajectory
```

#### Step 6: Check Results
```bash
# View output files
ls -la results/*my_trajectory*

# Check trajectory statistics
wc -l results/f_my_trajectory.txt  # Full trajectory
wc -l results/kf_my_trajectory.txt # Keyframes only

# View first few poses (TUM format: timestamp x y z qx qy qz qw)
head -5 results/f_my_trajectory.txt
```

**Expected Output:**
- `f_<name>.txt`: Frame trajectory with all tracked frames
- `kf_<name>.txt`: Keyframe trajectory with selected keyframes
- Typically tracks 70-80% of frames (e.g., 233/301 frames)
- Output format: TUM trajectory (timestamp + 7-DOF pose)

### Quick Test (All-in-One)
```bash
# Complete test pipeline
cd /home/external/ORB_SLAM3_AEA
source ~/venv/py39/bin/activate
source setup_env.sh

# Find and convert a test sequence
VRS_FILE=$(find /mnt/ssd_ext/incSeg-data/aria_everyday -name "*.vrs" | head -1)
python aria_to_tumvi.py "$VRS_FILE" test_data --duration 30

# Run ORB-SLAM3
./run_orbslam3_aria_tumvi_headless.sh test_data test_run

# Check results
echo "Poses tracked: $(wc -l < results/f_test_run.txt)"
echo "Keyframes: $(wc -l < results/kf_test_run.txt)"
```

## Overview

- **Camera**: SLAM left camera (640×480 @ 10Hz, 150° FOV, global shutter)
- **IMU**: Right IMU (1000Hz native, downsampled to 100Hz)
- **Processing**: 90° clockwise rotation applied to all images
- **Mode**: Monocular-Inertial (most reliable for Aria)

## Prerequisites

1. Python environment with projectaria_tools:
```bash
source ~/venv/py39/bin/activate
pip install projectaria_tools opencv-python numpy
```

2. Build ORB-SLAM3 (if not already built):
```bash
cd /home/external/ORB_SLAM3_AEA
./build.sh
```

## Working Implementation: Monocular-Inertial with TUM-VI Format

### Data Conversion Pipeline (aria_to_tumvi.py)
Converts Aria VRS files to TUM-VI format that ORB-SLAM3 understands:

```bash
# Full conversion
python aria_to_tumvi.py /path/to/recording.vrs output_dir

# Test with 30 seconds
python aria_to_tumvi.py /path/to/recording.vrs output_dir --duration 30 --imu-freq 100
```

Features:
- Extracts SLAM left camera (1201-1) at 10Hz
- Applies 90° clockwise rotation automatically
- Downsamples IMU from 1000Hz to 100Hz with interpolation
- Creates IMU measurements between camera frames (critical for preintegration)
- Saves nanosecond timestamps (ORB-SLAM3 requirement)

### Headless ORB-SLAM3 Execution
The `mono_inertial_tum_vi_noviewer` executable runs without display:

```bash
# Using the convenience script
./run_orbslam3_aria_tumvi_headless.sh output_dir trajectory_name

# Or directly
./Examples/Monocular-Inertial/mono_inertial_tum_vi_noviewer \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
    output_dir/mav0/cam0/data \
    output_dir/mav0/timestamps.txt \
    output_dir/mav0/imu0/data.csv \
    trajectory_name
```

### Configuration (Aria2TUM-VI.yaml)
- **Camera Model**: KannalaBrandt8 (fisheye)
- **Resolution**: 480×640 (after rotation)
- **Features**: 2000 ORB features
- **IMU Frequency**: 1000Hz in config (data is 100Hz)

### Output Files
- `f_<name>.txt`: Full frame trajectory (TUM format)
- `kf_<name>.txt`: Keyframe trajectory
- Example: 275 poses tracked from 301 frames

## Evaluating Performance

### Quick Analysis
```bash
# Analyze trajectory statistics
python evaluate_trajectory.py results/f_my_trajectory.txt
```

Expected good performance metrics:
- **Tracking rate**: > 80% of frames (e.g., 254/301 = 84%)
- **Average speed**: Consistent with motion (0.5-1.5 m/s for walking)
- **Large jumps**: < 2% of frames with jumps > 0.5m
- **Map resets**: Normal during initialization, system should recover

### Understanding "Fail to track local map!"
This is **NOT an error** - it's ORB-SLAM3's recovery mechanism:
- Common during initialization (first 10-15 seconds)
- System automatically creates new map and continues
- Look for "VIBA" messages indicating successful IMU integration

## Troubleshooting

1. **"Hanging" after loading**: Not actually hanging - vocabulary loading takes ~30 seconds
2. **No output progress**: Add debug prints or use the noviewer version with progress tracking
3. **"SLAM cameras not found"**: Check if the VRS file contains SLAM camera streams
4. **Tracking failures**: Ensure sufficient lighting and texture in the scene
5. **IMU initialization**: Need ~15 seconds of motion at startup
6. **Memory issues**: Use `--max-frames` to limit sequence length
7. **Timestamps in output**: Output timestamps are in nanoseconds (not seconds) - this is the Aria native format
8. **"Fail to track local map!"**: Normal recovery behavior, not an error

## Key Files Created

- `aria_to_tumvi.py` - VRS to TUM-VI converter with IMU interpolation
- `mono_inertial_tum_vi_noviewer.cc` - Headless ORB-SLAM3 executable
- `Examples/Monocular-Inertial/Aria2TUM-VI.yaml` - Aria camera/IMU configuration
- `run_orbslam3_aria_tumvi_headless.sh` - Convenience execution script
- `evaluate_trajectory.py` - Trajectory analysis tool
- `extract_ground_truth.py` - MPS ground truth extraction (if available)

## Notes

- The 90° rotation is critical - Aria cameras are mounted sideways
- Monocular-Inertial mode works more reliably than Stereo-Inertial
- IMU downsampling to 100Hz is essential for performance
- Vocabulary loading takes ~30 seconds (139MB file)
- The 150° FOV provides excellent tracking robustness
- IMU helps during fast motions and low-texture areas