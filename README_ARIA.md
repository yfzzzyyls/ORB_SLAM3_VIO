# Aria Digital Twin (ADT) Dataset for ORB-SLAM3

This guide explains how to process and run Aria Digital Twin (ADT) dataset with ORB-SLAM3.

## Successfully Running ORB-SLAM3 on Aria Data (2025-07-01)

### Key Achievements
1. **Data Conversion**: Successfully converts Aria VRS to TUM-VI format
   - 480×640 images with 90° rotation applied
   - IMU at 1000Hz (native rate, no downsampling)
   - Proper nanosecond timestamps

3. **Results**:
   - Map created with 313 points and 42 keyframes
   - Median tracking time: 13.9ms
   - Full trajectory saved (275 poses)
   - IMU preintegration working (confirmed by "VIBA" messages)

### Complete Step-by-Step Instructions

#### Step 1: Setup Environment
```bash
# Navigate to ORB-SLAM3 directory
cd /home/external/ORB_SLAM3_VIO

# Activate conda environment for Aria tools and visualization
source /home/external/miniconda3/bin/activate
conda activate orbslam

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

#### Step 3: Find an ADT VRS File
```bash
# Option 1: List all available VRS files to choose from
ls /mnt/ssd_ext/incSeg-data/adt/*/*main_recording.vrs

# Option 2: Find a specific sequence (e.g., seq131)
VRS_FILE=$(find /mnt/ssd_ext/incSeg-data/adt -name "*main_recording.vrs" | grep "seq131" | head -1)
echo "Found: $VRS_FILE"

# Option 3: Use a specific known path
VRS_FILE="/mnt/ssd_ext/incSeg-data/adt/Apartment_release_clean_seq131_M1292/ADT_Apartment_release_clean_seq131_M1292_main_recording.vrs"

# Option 4: Interactive selection
echo "Available sequences:"
ls /mnt/ssd_ext/incSeg-data/adt/
# Then manually set the path (replace Apartment_release_clean_seq131_M1292 with your chosen sequence)
VRS_FILE="/mnt/ssd_ext/incSeg-data/adt/Apartment_release_clean_seq131_M1292/ADT_Apartment_release_clean_seq131_M1292_main_recording.vrs"
```

#### Step 4: Convert VRS to TUM-VI Format
```bash
# Convert full sequence (recommended for complete evaluation)
python aria_to_tumvi.py "$VRS_FILE" aria_tumvi_full

# Or for quick testing with 30 seconds of data
python aria_to_tumvi.py "$VRS_FILE" aria_tumvi_test --duration 30
```

#### Step 5: Run ORB-SLAM3
```bash
# For full sequence (after converting with aria_tumvi_full)
./run_orbslam3_aria_tumvi.sh aria_tumvi_full my_trajectory

# For test sequence (if using aria_tumvi_test)
./run_orbslam3_aria_tumvi.sh aria_tumvi_test my_trajectory

# Or run directly
./Examples/Monocular-Inertial/mono_inertial_tum_vi \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
    aria_tumvi_full/mav0/cam0/data \
    aria_tumvi_full/mav0/timestamps.txt \
    aria_tumvi_full/mav0/imu0/data.csv \
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

#### Step 7: Evaluate SLAM Performance

**Important**: ADT dataset includes synthetic ground truth data (depth maps and trajectories):
```bash
# ADT sequences include ground truth in the same directory
# Check for available ground truth data:
ls /mnt/ssd_ext/incSeg-data/adt/Apartment_release_clean_seq131_M1292/

# Note: ADT provides synthetic ground truth rather than MPS trajectories
# The evaluation scripts may need modification for ADT's ground truth format
```

Then run the evaluation:
```bash
# Run complete evaluation with metrics and visualization plots
# For full sequence
./evaluate_slam_clean.sh aria_tumvi_full my_trajectory

# For test sequence (30 seconds)
./evaluate_slam_clean.sh aria_tumvi_test my_trajectory

# View results
cd evaluation && ls -la
```

**Generated Evaluation Files:**
- `ate_plot.pdf`: Absolute Trajectory Error visualization
- `rpe_1s_plot.pdf`: Relative Pose Error at 1 second intervals
- `rpe_5s_plot.pdf`: Relative Pose Error at 5 second intervals
- `slam_evaluation_summary.md`: Summary report with all metrics

**Note**: ADT provides synthetic ground truth data including depth maps and camera trajectories, which differs from the MPS trajectories in Aria Everyday Activities dataset.

## Notes

- The 90° rotation is critical - Aria cameras are mounted sideways
- Monocular-Inertial mode works more reliably than Stereo-Inertial
- IMU data is extracted at native 1000Hz rate for maximum accuracy
- Vocabulary loading takes ~30 seconds (139MB file)
- The 150° FOV provides excellent tracking robustness
- IMU helps during fast motions and low-texture areas