#!/usr/bin/env python3
"""
Convert Aria VRS files to TUM-VI format for ORB-SLAM3
Based on analysis of working implementation
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Check if projectaria_tools is available
try:
    from projectaria_tools.core import data_provider, calibration
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core.mps.utils import get_nearest_pose
except ImportError:
    print("Error: projectaria_tools not found. Please install with:")
    print("pip install projectaria-tools")
    sys.exit(1)


def convert_aria_to_tumvi(vrs_path, output_dir, start_time=0, duration=None):
    """
    Convert Aria VRS file to TUM-VI format
    
    Args:
        vrs_path: Path to Aria VRS file
        output_dir: Output directory for TUM-VI format data
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
    """
    print(f"Converting Aria VRS to TUM-VI format...")
    print(f"Input: {vrs_path}")
    print(f"Output: {output_dir}")
    print(f"IMU frequency: 1000 Hz (native)")
    
    # Create output directories
    output_path = Path(output_dir)
    mav0_path = output_path / "mav0"
    
    # Create directory structure
    (mav0_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (mav0_path / "cam1" / "data").mkdir(parents=True, exist_ok=True)
    (mav0_path / "imu0").mkdir(parents=True, exist_ok=True)
    
    # Load VRS data provider
    print("Loading VRS file...")
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        print(f"Error: Cannot open VRS file: {vrs_path}")
        return False
    
    # Get stream IDs
    camera_left_label = "camera-slam-left"   # 1201-1
    camera_right_label = "camera-slam-right" # 1201-2
    imu_label = "imu-right"                  # 1202-1
    
    camera_left_id = provider.get_stream_id_from_label(camera_left_label)
    camera_right_id = provider.get_stream_id_from_label(camera_right_label)
    imu_id = provider.get_stream_id_from_label(imu_label)
    
    # Get calibration
    camera_left_calib = provider.get_device_calibration().get_camera_calib(camera_left_label)
    camera_right_calib = provider.get_device_calibration().get_camera_calib(camera_right_label)
    
    # Get time range
    t_start_ns = provider.get_first_time_ns(camera_left_id, TimeDomain.DEVICE_TIME)
    t_end_ns = provider.get_last_time_ns(camera_left_id, TimeDomain.DEVICE_TIME)
    
    # Apply start time and duration if specified
    if start_time > 0:
        t_start_ns += int(start_time * 1e9)
    if duration:
        t_end_ns = min(t_end_ns, t_start_ns + int(duration * 1e9))
    
    print(f"Time range: {(t_end_ns - t_start_ns) / 1e9:.2f} seconds")
    
    # Process left camera images and collect timestamps
    print("\nExtracting SLAM camera images...")
    left_img_timestamps = []
    
    # Process left camera
    left_data_csv = mav0_path / "cam0" / "data.csv"
    with open(left_data_csv, 'w') as f:
        f.write("#timestamp [ns],filename\n")
        
        for idx in range(provider.get_num_data(camera_left_id)):
            img_data, img_info = provider.get_image_data_by_index(camera_left_id, idx)
            timestamp_ns = img_info.capture_timestamp_ns
            
            if timestamp_ns < t_start_ns:
                continue
            if timestamp_ns > t_end_ns:
                break
            
            left_img_timestamps.append(timestamp_ns)
            
            # Save image with 90° clockwise rotation
            image_array = img_data.to_numpy_array()
            rotated_image = np.rot90(image_array, k=3)  # 90° clockwise
            
            filename = f"{timestamp_ns}.png"
            filepath = mav0_path / "cam0" / "data" / filename
            Image.fromarray(rotated_image).save(filepath)
            f.write(f"{timestamp_ns},{filename}\n")
            
            if len(left_img_timestamps) % 100 == 0:
                print(f"  Processed {len(left_img_timestamps)} left camera images...")
    
    print(f"  Extracted {len(left_img_timestamps)} left camera images")
    
    # Process right camera - synchronized with left timestamps
    right_data_csv = mav0_path / "cam1" / "data.csv"
    with open(right_data_csv, 'w') as f:
        f.write("#timestamp [ns],filename\n")
        
        for timestamp_ns in left_img_timestamps:
            # Get closest right image to this timestamp
            img_data, img_info = provider.get_image_data_by_time_ns(
                camera_right_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
            )
            
            # Save with same timestamp as left for synchronization
            image_array = img_data.to_numpy_array()
            rotated_image = np.rot90(image_array, k=3)  # 90° clockwise
            
            filename = f"{timestamp_ns}.png"
            filepath = mav0_path / "cam1" / "data" / filename
            Image.fromarray(rotated_image).save(filepath)
            f.write(f"{timestamp_ns},{filename}\n")
            
            if len(left_img_timestamps) % 100 == 0 and left_img_timestamps[-1] == timestamp_ns:
                print(f"  Processed {len(left_img_timestamps)} right camera images...")
    
    print(f"  Extracted {len(left_img_timestamps)} right camera images")
    
    # Save timestamps for ORB-SLAM3 (nanoseconds)
    timestamps_path = mav0_path / "timestamps.txt"
    with open(timestamps_path, 'w') as f:
        for timestamp_ns in left_img_timestamps:
            f.write(f"{timestamp_ns}\n")
    
    print(f"\nCreated timestamps.txt with {len(left_img_timestamps)} entries")
    
    # Extract IMU data at native rate
    print(f"\nExtracting IMU data at native 1000Hz...")
    
    # For 1kHz IMU, we need to extract all IMU samples between the first and last camera timestamps
    # This ensures ORB-SLAM3 has IMU data available between consecutive camera frames
    
    # Write IMU data
    imu_data_path = mav0_path / "imu0" / "data.csv"
    imu_count = 0
    
    with open(imu_data_path, 'w') as f:
        f.write("#timestamp [ns],w_x [rad/s],w_y [rad/s],w_z [rad/s],a_x [m/s^2],a_y [m/s^2],a_z [m/s^2]\n")
        
        # Get all IMU samples in the time range
        for idx in range(provider.get_num_data(imu_id)):
            imu_data = provider.get_imu_data_by_index(imu_id, idx)
            timestamp_ns = imu_data.capture_timestamp_ns
            
            # Skip samples outside our time range
            if timestamp_ns < t_start_ns:
                continue
            if timestamp_ns > t_end_ns:
                break
            
            # Extract gyro and accel data
            gx, gy, gz = imu_data.gyro_radsec
            ax, ay, az = imu_data.accel_msec2
            
            f.write(f"{timestamp_ns},{gx},{gy},{gz},{ax},{ay},{az}\n")
            imu_count += 1
            
            if imu_count % 10000 == 0:
                print(f"  Processed {imu_count} IMU samples...")
    
    print(f"  Extracted {imu_count} IMU samples at 1000Hz")
    
    # Calculate and save IMU-Camera transformation
    print("\nCalculating IMU-Camera transformation...")
    save_transforms(provider, mav0_path)
    
    print("\nConversion complete!")
    print(f"\nTo run ORB-SLAM3:")
    print(f"./Examples/Monocular-Inertial/mono_inertial_tum_vi \\")
    print(f"  Vocabulary/ORBvoc.txt \\")
    print(f"  Examples/Monocular-Inertial/Aria2TUM-VI.yaml \\")
    print(f"  {mav0_path}/cam0/data \\")
    print(f"  {mav0_path}/timestamps.txt \\")
    print(f"  {mav0_path}/imu0/data.csv \\")
    print(f"  output_trajectory")
    
    return True


def save_transforms(provider, save_path):
    """Calculate and save IMU-Camera transformation matrices"""
    camera_left_label = "camera-slam-left"
    camera_right_label = "camera-slam-right"
    
    # Get calibrations
    camera_left_calib = provider.get_device_calibration().get_camera_calib(camera_left_label)
    camera_right_calib = provider.get_device_calibration().get_camera_calib(camera_right_label)
    
    # Rotate calibrations by 90° clockwise
    left_cw90 = calibration.rotate_camera_calib_cw90deg(camera_left_calib)
    right_cw90 = calibration.rotate_camera_calib_cw90deg(camera_right_calib)
    
    imu_calib = provider.get_device_calibration().get_imu_calib("imu-right")
    
    if not all([camera_left_calib, camera_right_calib, imu_calib]):
        print("Warning: Cannot get calibration data")
        return
    
    # Get transforms
    T_device_camera_left = left_cw90.get_transform_device_camera()
    T_device_camera_right = right_cw90.get_transform_device_camera()
    T_device_imu = imu_calib.get_transform_device_imu()
    
    # Calculate transformations
    T_imu_camera_left = T_device_imu.inverse() @ T_device_camera_left
    T_camera_left_right = T_device_camera_left.inverse() @ T_device_camera_right
    
    # Convert to matrices
    T_b_c1 = T_imu_camera_left.to_matrix()
    T_c1_c2 = T_camera_left_right.to_matrix()
    
    print(f"T_b_c1 (IMU to left camera):\n{T_b_c1}")
    print(f"\nT_c1_c2 (left to right camera):\n{T_c1_c2}")
    
    # Save to file
    transforms_file = save_path / "calibration_info.txt"
    with open(transforms_file, 'w') as f:
        f.write("# Transformation matrices for ORB-SLAM3\n")
        f.write("# Generated by aria_to_tumvi.py\n\n")
        f.write(f"T_b_c1 (IMU to left camera):\n{T_b_c1}\n\n")
        f.write(f"T_c1_c2 (left to right camera):\n{T_c1_c2}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert Aria VRS to TUM-VI format')
    parser.add_argument('vrs_file', help='Path to input VRS file')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('--start-time', type=float, default=0, 
                        help='Start time in seconds (optional)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (optional)')
    
    args = parser.parse_args()
    
    # Check if VRS file exists
    if not os.path.exists(args.vrs_file):
        print(f"Error: VRS file not found: {args.vrs_file}")
        sys.exit(1)
    
    # Convert
    success = convert_aria_to_tumvi(
        args.vrs_file,
        args.output_dir,
        args.start_time,
        args.duration
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()