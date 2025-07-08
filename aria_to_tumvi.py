#!/usr/bin/env python3
"""
Convert Aria Digital Twin (ADT) VRS files to TUM-VI format for ORB-SLAM3
Optimized for monocular-inertial SLAM with left SLAM camera only
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

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


def convert_aria_to_tumvi(vrs_path, output_dir, start_time=0, duration=None, rectify=False):
    """
    Convert Aria Digital Twin (ADT) VRS file to TUM-VI format for monocular-inertial SLAM
    
    Args:
        vrs_path: Path to ADT VRS file
        output_dir: Output directory for TUM-VI format data
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        rectify: If True, rectify fisheye images to pinhole projection (default: False)
    """
    print(f"Converting Aria Digital Twin (ADT) VRS to TUM-VI format...")
    print(f"Input: {vrs_path}")
    print(f"Output: {output_dir}")
    print(f"Camera: Left SLAM camera @ 30 Hz")
    print(f"IMU: Right IMU @ 1000 Hz (native)")
    
    # Create output directories (monocular only)
    output_path = Path(output_dir)
    mav0_path = output_path / "mav0"
    
    # Create directory structure for monocular setup
    (mav0_path / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (mav0_path / "imu0").mkdir(parents=True, exist_ok=True)
    
    # Load VRS data provider
    print("\nLoading VRS file...")
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        print(f"Error: Cannot open VRS file: {vrs_path}")
        return False
    
    # Get stream IDs (only left SLAM camera and right IMU)
    camera_label = "camera-slam-left"   # 1201-1
    imu_label = "imu-right"             # 1202-1
    
    camera_id = provider.get_stream_id_from_label(camera_label)
    imu_id = provider.get_stream_id_from_label(imu_label)
    
    # Get calibration
    camera_calib = provider.get_device_calibration().get_camera_calib(camera_label)
    
    # Create pinhole calibration if rectifying
    pinhole_calib = None
    if rectify:
        print("Using pinhole rectification (512x512, fx=fy=150)...")
        pinhole_calib = calibration.get_linear_camera_calibration(
            512, 512, 150.0,
            camera_label,
            camera_calib.get_transform_device_camera()
        )
    
    # Get time range
    t_start_ns = provider.get_first_time_ns(camera_id, TimeDomain.DEVICE_TIME)
    t_end_ns = provider.get_last_time_ns(camera_id, TimeDomain.DEVICE_TIME)
    
    # Apply start time and duration if specified
    if start_time > 0:
        t_start_ns += int(start_time * 1e9)
    if duration:
        t_end_ns = min(t_end_ns, t_start_ns + int(duration * 1e9))
    
    print(f"Time range: {(t_end_ns - t_start_ns) / 1e9:.2f} seconds")
    
    # Process camera images and collect timestamps
    print("\nExtracting left SLAM camera images (30 Hz)...")
    img_timestamps = []
    
    # Process camera images
    data_csv = mav0_path / "cam0" / "data.csv"
    with open(data_csv, 'w') as f:
        f.write("#timestamp [ns],filename\n")
        
        for idx in range(provider.get_num_data(camera_id)):
            img_data, img_info = provider.get_image_data_by_index(camera_id, idx)
            timestamp_ns = img_info.capture_timestamp_ns
            
            if timestamp_ns < t_start_ns:
                continue
            if timestamp_ns > t_end_ns:
                break
            
            img_timestamps.append(timestamp_ns)
            
            # Get image array
            image_array = img_data.to_numpy_array()
            
            # Rectify if requested
            if rectify:
                image_array = calibration.distort_by_calibration(
                    image_array, pinhole_calib, camera_calib
                )
            
            # Rotate 90° clockwise (Aria cameras are mounted sideways)
            rotated_image = np.rot90(image_array, k=3)
            
            filename = f"{timestamp_ns}.png"
            filepath = mav0_path / "cam0" / "data" / filename
            Image.fromarray(rotated_image).save(filepath)
            f.write(f"{timestamp_ns},{filename}\n")
            
            if len(img_timestamps) % 100 == 0:
                print(f"  Processed {len(img_timestamps)} images...")
    
    print(f"  Extracted {len(img_timestamps)} camera images @ 30 Hz")
    
    # Save timestamps for ORB-SLAM3 (nanoseconds)
    timestamps_path = mav0_path / "timestamps.txt"
    with open(timestamps_path, 'w') as f:
        for timestamp_ns in img_timestamps:
            f.write(f"{timestamp_ns}\n")
    
    print(f"\nCreated timestamps.txt with {len(img_timestamps)} entries")
    
    # Extract IMU data at native rate
    print(f"\nExtracting IMU data at native 1000 Hz...")
    
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
    
    print(f"  Extracted {imu_count} IMU samples")
    
    # Calculate average IMU rate
    if len(img_timestamps) > 1:
        camera_duration = (img_timestamps[-1] - img_timestamps[0]) / 1e9
        avg_imu_rate = imu_count / camera_duration
        print(f"  Average IMU rate: {avg_imu_rate:.1f} Hz")
    
    # Calculate and save IMU-Camera transformation
    print("\nCalculating IMU-Camera transformation...")
    save_transforms(provider, mav0_path, rectify)
    
    print("\nConversion complete!")
    print(f"\nTo run ORB-SLAM3 in monocular-inertial mode:")
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
    
    # Create dataset.yaml with ADT-specific information
    dataset_yaml = output_path / "dataset.yaml"
    # Extract sequence name from VRS path for ADT format
    vrs_path_parts = Path(vrs_path).parts
    sequence_name = None
    for part in vrs_path_parts:
        if part.startswith("Apartment_release_clean_seq"):
            sequence_name = part
            break
    
    with open(dataset_yaml, 'w') as f:
        f.write("%YAML:1.0\n")
        f.write(f"dataset_name: Aria Digital Twin (ADT)\n")
        if sequence_name:
            f.write(f"sequence_name: {sequence_name}\n")
        f.write(f"vrs_file: {vrs_path}\n")
        f.write(f"camera_rate: 30.0\n")  # ADT SLAM cameras run at 30 Hz
        f.write(f"imu_rate: 1000.0\n")
        f.write(f"duration: {(t_end_ns - t_start_ns) / 1e9:.1f}\n")
        f.write(f"num_images: {len(img_timestamps)}\n")
        f.write(f"camera_type: {'pinhole' if rectify else 'fisheye'}\n")
        f.write(f"camera_stream: camera-slam-left\n")
    
    return True


def save_transforms(provider, save_path, rectify=False):
    """Calculate and save IMU-Camera transformation matrix for monocular setup"""
    camera_label = "camera-slam-left"
    
    # Get calibrations
    camera_calib = provider.get_device_calibration().get_camera_calib(camera_label)
    
    # Handle calibration based on rectification mode
    if rectify:
        # For rectified images, use pinhole calibration
        camera_cw90 = calibration.get_linear_camera_calibration(
            512, 512, 150.0,
            camera_label,
            camera_calib.get_transform_device_camera()
        )
        # Rotate the pinhole calibration
        camera_cw90 = calibration.rotate_camera_calib_cw90deg(camera_cw90)
    else:
        # For raw images, just rotate the fisheye calibration
        camera_cw90 = calibration.rotate_camera_calib_cw90deg(camera_calib)
    
    imu_calib = provider.get_device_calibration().get_imu_calib("imu-right")
    
    if not all([camera_calib, imu_calib]):
        print("Warning: Cannot get calibration data")
        return
    
    # Get transforms
    T_device_camera = camera_cw90.get_transform_device_camera()
    T_device_imu = imu_calib.get_transform_device_imu()
    
    # Calculate IMU to camera transformation
    T_imu_camera = T_device_imu.inverse() @ T_device_camera
    
    # Convert to matrix
    T_bc = T_imu_camera.to_matrix()
    
    print(f"T_bc (IMU to camera):\n{T_bc}")
    
    # Save to file
    transforms_file = save_path / "calibration_info.txt"
    with open(transforms_file, 'w') as f:
        f.write("# Transformation matrices for ORB-SLAM3 monocular-inertial\n")
        f.write("# Generated by aria_to_tumvi.py for ADT dataset\n\n")
        f.write(f"T_bc (IMU to camera):\n{T_bc}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Aria Digital Twin (ADT) VRS to TUM-VI format for monocular-inertial SLAM'
    )
    parser.add_argument('vrs_file', help='Path to ADT VRS file')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('--start-time', type=float, default=0, 
                        help='Start time in seconds (optional)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (optional)')
    parser.add_argument('--rectify', action='store_true',
                        help='Rectify fisheye images to pinhole projection (512x512)')
    
    args = parser.parse_args()
    
    # Check if VRS file exists
    if not os.path.exists(args.vrs_file):
        print(f"Error: VRS file not found: {args.vrs_file}")
        sys.exit(1)
    
    # Verify it's an ADT file
    if "adt" not in str(args.vrs_file).lower() and "apartment_release" not in str(args.vrs_file).lower():
        print("Warning: This doesn't appear to be an ADT VRS file.")
        print("Expected path containing 'adt' or 'Apartment_release'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Convert
    success = convert_aria_to_tumvi(
        args.vrs_file,
        args.output_dir,
        args.start_time,
        args.duration,
        args.rectify
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()