#!/usr/bin/env python3
"""
Convert processed ADT data (from extract_adt.py) to TUM-VI format for ORB-SLAM3.
This script reads the extracted raw data and applies necessary transformations:
- Keeps SLAM images in original 640x480 orientation
- Formats IMU data for ORB-SLAM3
- Creates proper TUM-VI directory structure
- Supports rectification to pinhole model
- Generates calibration info and dataset metadata
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import shutil

# Check if projectaria_tools is available for calibration functions
try:
    from projectaria_tools.core import calibration
    from projectaria_tools.core.calibration import CameraCalibration
    PROJECTARIA_AVAILABLE = True
except ImportError:
    PROJECTARIA_AVAILABLE = False
    print("Warning: projectaria_tools not found. Rectification features will be limited.")
    print("Install with: pip install projectaria-tools")



def load_camera_calibration(calib_json_path, camera_name='slam_left'):
    """Load camera calibration from extracted JSON file."""
    with open(calib_json_path, 'r') as f:
        calib_data = json.load(f)
    
    if camera_name not in calib_data['cameras']:
        return None
    
    return calib_data['cameras'][camera_name]


def calculate_T_bc(T_device_camera, T_device_imu):
    """
    Calculate T_bc (IMU to camera transform) from device transforms.
    
    Args:
        T_device_camera: 4x4 transform from device to camera
        T_device_imu: 4x4 transform from device to IMU
        
    Returns:
        T_bc: 4x4 transform from IMU (body) to camera
    """
    # Calculate T_bc = T_imu_camera = inv(T_device_imu) @ T_device_camera
    T_imu_device = np.linalg.inv(T_device_imu)
    T_bc = T_imu_device @ T_device_camera
    
    return T_bc


def create_fisheye_camera_calibration(calib_data):
    """Create a CameraCalibration object from extracted calibration data."""
    if not PROJECTARIA_AVAILABLE:
        return None
    
    try:
        # Create calibration based on model type
        model_name = calib_data['model_name']
        
        if 'FISHEYE' in model_name or 'KANNALA' in model_name:
            # For fisheye models, use the projection parameters
            if calib_data['projection_params']:
                camera_calib = calibration.get_kb4_camera_calibration(
                    calib_data['image_width'],
                    calib_data['image_height'],
                    calib_data['focal_lengths'][0],
                    calib_data['focal_lengths'][1],
                    calib_data['principal_point'][0],
                    calib_data['principal_point'][1],
                    calib_data['distortion_coeffs'][0] if len(calib_data['distortion_coeffs']) > 0 else 0.0,
                    calib_data['distortion_coeffs'][1] if len(calib_data['distortion_coeffs']) > 1 else 0.0,
                    calib_data['distortion_coeffs'][2] if len(calib_data['distortion_coeffs']) > 2 else 0.0,
                    calib_data['distortion_coeffs'][3] if len(calib_data['distortion_coeffs']) > 3 else 0.0
                )
            else:
                # Fallback to linear model
                camera_calib = calibration.get_linear_camera_calibration(
                    calib_data['image_width'],
                    calib_data['image_height'],
                    calib_data['focal_lengths'][0],
                    calib_data['label']
                )
        else:
            # Linear/pinhole model
            camera_calib = calibration.get_linear_camera_calibration(
                calib_data['image_width'],
                calib_data['image_height'],
                calib_data['focal_lengths'][0],
                calib_data['label']
            )
        
        return camera_calib
    except Exception as e:
        print(f"Warning: Could not create camera calibration: {e}")
        return None


def rectify_fisheye_image(image, calib_data=None, rectified_size=512):
    """
    Rectify fisheye image to pinhole model.
    """
    if PROJECTARIA_AVAILABLE and calib_data is not None:
        try:
            # Create fisheye calibration
            fisheye_calib = create_fisheye_camera_calibration(calib_data)
            if fisheye_calib:
                # Create target pinhole calibration
                pinhole_calib = calibration.get_linear_camera_calibration(
                    rectified_size, rectified_size, 150.0,
                    calib_data['label']
                )
                
                # Rectify image
                rectified = calibration.distort_by_calibration(
                    image, pinhole_calib, fisheye_calib
                )
                return rectified
        except Exception as e:
            print(f"Warning: Rectification failed, using fallback: {e}")
    
    # Fallback: Simple center crop and resize
    h, w = image.shape[:2]
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
    return cv2.resize(cropped, (rectified_size, rectified_size))


def convert_sequence_to_tumvi(seq_dir, output_dir, start_time=0, duration=None, rectify=False, use_left_camera=False):
    """
    Convert a single processed ADT sequence to TUM-VI format.
    
    Args:
        seq_dir: Path to processed sequence directory
        output_dir: Output directory for TUM-VI format
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        rectify: If True, rectify fisheye images to pinhole projection
        use_left_camera: If True, use left SLAM camera instead of right (default: right)
    """
    seq_name = seq_dir.name
    print(f"\nConverting {seq_name} to TUM-VI format...")
    print(f"Camera: {'Left' if use_left_camera else 'Right'} SLAM camera @ 30 Hz")
    print(f"IMU: {'Left' if use_left_camera else 'Right'} IMU @ {'800' if use_left_camera else '1000'} Hz (native)")
    print(f"Image format: Original 640x480 orientation")
    if rectify:
        print(f"Rectification: Enabled (512x512 pinhole)")
    
    # Check if sequence has required data
    slam_right_dir = seq_dir / "slam_right"
    slam_left_dir = seq_dir / "slam_left"
    imu_right_file = seq_dir / "imu" / "imu_data.json"  # Right IMU (1000Hz)
    imu_left_file = seq_dir / "imu" / "imu_left_data.json"  # Left IMU (~800Hz)
    metadata_file = seq_dir / "metadata.json"
    calibration_file = seq_dir / "calibration" / "calibration.json"
    
    # Select camera and IMU based on parameter (default: right side for both)
    if use_left_camera:
        slam_dir = slam_left_dir
        imu_file = imu_left_file
        timestamps_file = seq_dir / "slam_left_timestamps.json"
        camera_label = 'slam-left'
        imu_label = 'left'
    else:
        slam_dir = slam_right_dir
        imu_file = imu_right_file
        timestamps_file = seq_dir / "slam_right_timestamps.json"
        camera_label = 'slam-right'
        imu_label = 'right'
    
    if not slam_dir.exists():
        print(f"Error: No {'left' if use_left_camera else 'right'} SLAM camera data found in {seq_dir}")
        if use_left_camera and slam_right_dir.exists():
            print(f"Note: Right SLAM camera data is available. Run without --use-left-camera to use it.")
        elif not use_left_camera and slam_left_dir.exists():
            print(f"Note: Left SLAM camera data is available. Run with --use-left-camera to use it.")
        return False
    
    if not imu_file.exists():
        print(f"Error: No {'left' if use_left_camera else 'right'} IMU data found at {imu_file}")
        return False
    
    # Load calibration data if available
    slam_calib_data = None
    if calibration_file.exists() and rectify:
        slam_calib_data = load_camera_calibration(calibration_file, camera_label)
        if slam_calib_data:
            print(f"  Loaded {camera_label} calibration for rectification")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load SLAM timestamps
    if not timestamps_file.exists():
        print(f"Error: No timestamps file found at {timestamps_file}")
        return False
    
    with open(timestamps_file, 'r') as f:
        slam_timestamps = json.load(f)
    
    # Create TUM-VI directory structure
    output_path = Path(output_dir)
    mav0_path = output_path / "mav0"
    (mav0_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (mav0_path / "imu0").mkdir(parents=True, exist_ok=True)
    
    # Load IMU data
    with open(imu_file, 'r') as f:
        imu_data = json.load(f)
    
    # Convert timestamps to seconds and apply time range
    slam_timestamps_ns = [ts['timestamp_ns'] for ts in slam_timestamps]
    slam_timestamps_s = [ts / 1e9 for ts in slam_timestamps_ns]
    
    # Apply start time and duration
    start_time_s = slam_timestamps_s[0] + start_time
    if duration:
        end_time_s = start_time_s + duration
    else:
        end_time_s = slam_timestamps_s[-1]
    
    # Find valid frame range
    valid_indices = []
    for i, ts in enumerate(slam_timestamps_s):
        if start_time_s <= ts <= end_time_s:
            valid_indices.append(i)
    
    if not valid_indices:
        print(f"Error: No frames in specified time range")
        return False
    
    print(f"Processing {len(valid_indices)} frames from {len(slam_timestamps)} total")
    
    # Create timestamps.txt file (single global timestamp file for monocular)
    timestamps_txt = mav0_path / "timestamps.txt"
    with open(timestamps_txt, 'w') as f:
        for idx in valid_indices:
            f.write(f"{slam_timestamps_ns[idx]}\n")
    
    # Process camera images
    print("Processing SLAM images...")
    cam_csv = mav0_path / "cam0" / "data.csv"
    
    with open(cam_csv, 'w') as f:
        f.write("#timestamp [ns],filename\n")
        
        for new_idx, orig_idx in enumerate(tqdm(valid_indices)):
            # Read original image
            orig_filename = f"frame_{orig_idx:06d}.png"
            orig_path = slam_dir / orig_filename
            
            if not orig_path.exists():
                print(f"Warning: Missing image {orig_filename}")
                continue
            
            # Load image (original orientation)
            img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Cannot read image {orig_filename}")
                continue
            
            # Apply rectification if requested
            if rectify:
                img_final = rectify_fisheye_image(img, slam_calib_data)
                # Use timestamp as filename for rectified images
                new_filename = f"{slam_timestamps_ns[orig_idx]}.png"
            else:
                img_final = img
                new_filename = f"{new_idx:06d}.png"
            
            # Save image
            new_path = mav0_path / "cam0" / "data" / new_filename
            cv2.imwrite(str(new_path), img_final)
            
            # Write to CSV
            f.write(f"{slam_timestamps_ns[orig_idx]},{new_filename}\n")
    
    # Process IMU data
    print("Processing IMU data...")
    imu_csv = mav0_path / "imu0" / "data.csv"
    
    # Filter IMU data by time range
    start_ns = slam_timestamps_ns[valid_indices[0]]
    end_ns = slam_timestamps_ns[valid_indices[-1]]
    
    imu_count = 0
    with open(imu_csv, 'w') as f:
        f.write("#timestamp [ns],w_x [rad/s],w_y [rad/s],w_z [rad/s],a_x [m/s^2],a_y [m/s^2],a_z [m/s^2]\n")
        
        for imu_sample in imu_data:
            ts_ns = imu_sample['timestamp_ns']
            if start_ns <= ts_ns <= end_ns:
                gyro = imu_sample['gyro']
                accel = imu_sample['accel']
                f.write(f"{ts_ns},{gyro[0]},{gyro[1]},{gyro[2]},{accel[0]},{accel[1]},{accel[2]}\n")
                imu_count += 1
                
                if imu_count % 10000 == 0:
                    print(f"  Processed {imu_count} IMU samples...")
    
    print(f"  Extracted {imu_count} IMU samples")
    
    # Calculate average IMU rate
    if len(valid_indices) > 1:
        camera_duration = (slam_timestamps_ns[valid_indices[-1]] - slam_timestamps_ns[valid_indices[0]]) / 1e9
        avg_imu_rate = imu_count / camera_duration
        print(f"  Average IMU rate: {avg_imu_rate:.1f} Hz")
    
    # Generate sensor.yaml based on rectification mode
    if rectify:
        # Pinhole parameters for rectified images
        camera_model = "pinhole"
        intrinsics = [150.0, 150.0, 256.0, 256.0]  # fx, fy, cx, cy for 512x512
        resolution = [512, 512]
        distortion_coeffs = [0.0, 0.0, 0.0, 0.0]
    else:
        # Original fisheye parameters (640x480)
        camera_model = "pinhole"  # Simplified for TUM-VI format
        # Use original calibration values
        intrinsics = [241.092481, 241.092481, 316.638312, 238.204572]
        resolution = [640, 480]
        distortion_coeffs = [0.0, 0.0, 0.0, 0.0]
    
    sensor_yaml_content = f"""# Sensor configuration for Aria Digital Twin (monocular-inertial)
# Generated by processed_adt_to_tumvi.py

# Camera calibration (original orientation)
cam0:
  T_cam_imu:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: []
  camera_model: {camera_model}
  distortion_coeffs: {distortion_coeffs}
  distortion_model: radtan
  intrinsics: {intrinsics}
  resolution: {resolution}
  rostopic: /cam0/image_raw

# IMU calibration ({'left' if use_left_camera else 'right'} IMU)
imu0:
  T_i_b:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.00308
  accelerometer_random_walk: 0.000113
  gyroscope_noise_density: 0.00018
  gyroscope_random_walk: 0.0000045
  model: calibrated
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 1000.0
"""
    
    sensor_yaml = mav0_path / "sensor.yaml"
    with open(sensor_yaml, 'w') as f:
        f.write(sensor_yaml_content)
    
    # Create calibration_info.txt with transformation matrices
    # Default T_bc (identity) - will be replaced if calibration available
    T_bc = np.eye(4)
    
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            T_device_cam = None
            T_device_imu = None
            
            # Get device-to-camera and device-to-IMU transforms
            if 'cameras' in calib_data and camera_label in calib_data['cameras']:
                T_device_cam = np.array(calib_data['cameras'][camera_label]['transform_device_camera'])
                print(f"  Loaded {camera_label} calibration from extracted data")
            
            # Get appropriate IMU transform based on selection
            if 'imus' in calib_data and imu_label in calib_data['imus']:
                T_device_imu = np.array(calib_data['imus'][imu_label]['transform_device_imu'])
                print(f"  Loaded {imu_label} calibration from extracted data")
            
            # Calculate T_bc if both transforms are available
            if T_device_cam is not None and T_device_imu is not None:
                T_bc = calculate_T_bc(T_device_cam, T_device_imu)
                print(f"  Calculated T_bc for {imu_label} to {camera_label}")
            else:
                print("  Warning: Missing transforms, using default T_bc")
                
        except Exception as e:
            print(f"Warning: Could not load transforms from calibration: {e}")
    
    calibration_info = mav0_path / "calibration_info.txt"
    with open(calibration_info, 'w') as f:
        f.write("# Transformation matrices for ORB-SLAM3 monocular-inertial\n")
        f.write("# Generated by processed_adt_to_tumvi.py for ADT dataset\n\n")
        f.write(f"T_bc (IMU to camera):\n{T_bc}\n")
    
    # Create dataset.yaml with ADT-specific information
    dataset_yaml = output_path / "dataset.yaml"
    actual_duration = (slam_timestamps_ns[valid_indices[-1]] - slam_timestamps_ns[valid_indices[0]]) / 1e9
    
    with open(dataset_yaml, 'w') as f:
        f.write("%YAML:1.0\n")
        f.write(f"dataset_name: Aria Digital Twin (ADT)\n")
        f.write(f"sequence_name: {seq_name}\n")
        f.write(f"camera_rate: 30.0\n")  # ADT SLAM cameras run at 30 Hz
        f.write(f"imu_rate: 1000.0\n")
        f.write(f"duration: {actual_duration:.1f}\n")
        f.write(f"num_images: {len(valid_indices)}\n")
        f.write(f"num_imu_samples: {imu_count}\n")
        f.write(f"camera_type: {'pinhole' if rectify else 'fisheye'}\n")
        f.write(f"camera_stream: camera-{camera_label}\n")
        f.write(f"imu_stream: {imu_label}\n")
        f.write(f"rectified: {'true' if rectify else 'false'}\n")
        f.write(f"rotated: false\n")
    
    print(f"\nConversion complete for {seq_name}")
    print(f"Output saved to: {output_path}")
    print(f"Duration: {actual_duration:.1f} seconds")
    print(f"Images: {len(valid_indices)}")
    print(f"IMU samples: {imu_count}")
    
    # Print ORB-SLAM3 command
    print("\nTo run ORB-SLAM3 in monocular-inertial mode:")
    print(f"./Examples/Monocular-Inertial/mono_inertial_tum_vi \\")
    print(f"  Vocabulary/ORBvoc.txt \\")
    
    if rectify:
        print(f"  Examples/Monocular-Inertial/Aria2TUM-VI_Pinhole.yaml \\")
    else:
        print(f"  Examples/Monocular-Inertial/Aria2TUM-VI.yaml \\")
    
    print(f"  {mav0_path}/cam0/data \\")
    print(f"  {mav0_path}/timestamps.txt \\")
    print(f"  {mav0_path}/imu0/data.csv \\")
    print(f"  output_trajectory")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert processed ADT data to TUM-VI format for monocular-inertial SLAM'
    )
    parser.add_argument('input', type=str,
                        help='Input processed ADT sequence directory or parent directory')
    parser.add_argument('output_dir', type=str,
                        help='Output directory for TUM-VI format')
    parser.add_argument('--start-time', type=float, default=0,
                        help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (default: full sequence)')
    parser.add_argument('--rectify', action='store_true',
                        help='Rectify fisheye images to pinhole projection (512x512)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Specific sequence name if input is parent directory')
    parser.add_argument('--use-left-camera', action='store_true',
                        help='Use left SLAM camera and left IMU instead of right (default: right)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine if input is a single sequence or parent directory
    if args.sequence:
        # User specified a sequence within a parent directory
        seq_dir = input_path / args.sequence
        if not seq_dir.exists():
            print(f"Error: Sequence directory not found: {seq_dir}")
            sys.exit(1)
        sequences = [seq_dir]
    elif (input_path / "metadata.json").exists():
        # Input is a single sequence directory
        sequences = [input_path]
    else:
        # Input is a parent directory, find all sequences
        sequences = []
        for item in sorted(input_path.iterdir()):
            if item.is_dir() and (item / "metadata.json").exists():
                sequences.append(item)
        
        if not sequences:
            print(f"Error: No valid sequences found in {input_path}")
            sys.exit(1)
    
    print(f"Found {len(sequences)} sequence(s) to convert")
    
    # Convert each sequence
    for seq_dir in sequences:
        seq_name = seq_dir.name
        output_dir = Path(args.output_dir) / seq_name
        
        success = convert_sequence_to_tumvi(
            seq_dir, 
            output_dir,
            args.start_time,
            args.duration,
            args.rectify,
            args.use_left_camera
        )
        
        if not success:
            print(f"Failed to convert {seq_name}")
    
    print("\nAll conversions complete!")


if __name__ == "__main__":
    main()