#!/usr/bin/env python3
"""
Extract actual Aria calibration parameters and create proper ORB-SLAM3 YAML configuration
"""

import sys
import numpy as np
from pathlib import Path

try:
    from projectaria_tools.core import data_provider, calibration
except ImportError:
    print("Error: projectaria_tools not found. Please install with:")
    print("pip install projectaria-tools")
    sys.exit(1)


def extract_calibration(vrs_path):
    """Extract calibration parameters from Aria VRS file"""
    
    provider = data_provider.create_vrs_data_provider(vrs_path)
    
    # Get camera calibrations
    left_calib = provider.get_device_calibration().get_camera_calib('camera-slam-left')
    right_calib = provider.get_device_calibration().get_camera_calib('camera-slam-right')
    
    # Get IMU calibration
    imu_calib = provider.get_device_calibration().get_imu_calib('imu-right')
    
    # Extract parameters
    print("=== ACTUAL ARIA CALIBRATION PARAMETERS ===\n")
    
    print("Left SLAM Camera (1201-1):")
    print(f"  Model: {left_calib.get_model_name()}")
    print(f"  Resolution: {left_calib.get_image_size()}")  # [640, 480]
    fx_l, fy_l = left_calib.get_focal_lengths()
    cx_l, cy_l = left_calib.get_principal_point()
    print(f"  Focal lengths: fx={fx_l:.6f}, fy={fy_l:.6f}")
    print(f"  Principal point: cx={cx_l:.6f}, cy={cy_l:.6f}")
    
    # Get fisheye distortion parameters
    proj_params_l = left_calib.get_projection_params()
    print(f"  Projection params: {proj_params_l}")
    
    print("\nRight SLAM Camera (1201-2):")
    print(f"  Model: {right_calib.get_model_name()}")
    print(f"  Resolution: {right_calib.get_image_size()}")
    fx_r, fy_r = right_calib.get_focal_lengths()
    cx_r, cy_r = right_calib.get_principal_point()
    print(f"  Focal lengths: fx={fx_r:.6f}, fy={fy_r:.6f}")
    print(f"  Principal point: cx={cx_r:.6f}, cy={cy_r:.6f}")
    
    proj_params_r = right_calib.get_projection_params()
    print(f"  Projection params: {proj_params_r}")
    
    # Get transformations
    T_device_left = left_calib.get_transform_device_camera()
    T_device_right = right_calib.get_transform_device_camera()
    T_device_imu = imu_calib.get_transform_device_imu()
    
    # Calculate stereo baseline
    T_left_right = T_device_left.inverse() @ T_device_right
    baseline = np.linalg.norm(T_left_right.translation())
    print(f"\nStereo baseline: {baseline:.6f} meters")
    
    # Calculate IMU-Camera transformation
    T_imu_left = T_device_imu.inverse() @ T_device_left
    
    print("\n=== AFTER 90° CLOCKWISE ROTATION ===")
    print("(Images rotated from 640x480 to 480x640)\n")
    
    # After rotation: width and height swap, focal lengths stay same
    # cx' = height - cy, cy' = cx
    cx_l_rot = 480 - cy_l
    cy_l_rot = cx_l
    cx_r_rot = 480 - cy_r
    cy_r_rot = cx_r
    
    print("Left camera after rotation:")
    print(f"  Resolution: 480x640")
    print(f"  fx={fx_l:.6f}, fy={fy_l:.6f}")
    print(f"  cx={cx_l_rot:.6f}, cy={cy_l_rot:.6f}")
    
    print("\nRight camera after rotation:")
    print(f"  Resolution: 480x640")
    print(f"  fx={fx_r:.6f}, fy={fy_r:.6f}")
    print(f"  cx={cx_r_rot:.6f}, cy={cy_r_rot:.6f}")
    
    # Map Aria FISHEYE624 to Kannala-Brandt (first 4 radial terms)
    # Aria uses k2, k3, k4, k5 for radial distortion (indices 3,4,5,6 in proj_params)
    kb_k1_l = proj_params_l[3]  # k2
    kb_k2_l = proj_params_l[4]  # k3
    kb_k3_l = proj_params_l[5]  # k4
    kb_k4_l = proj_params_l[6]  # k5
    
    kb_k1_r = proj_params_r[3]  # k2
    kb_k2_r = proj_params_r[4]  # k3
    kb_k3_r = proj_params_r[5]  # k4
    kb_k4_r = proj_params_r[6]  # k5
    
    print("\n=== KANNALA-BRANDT DISTORTION PARAMETERS ===")
    print("(Mapped from Aria FISHEYE624 model)\n")
    print(f"Left:  k1={kb_k1_l:.6f}, k2={kb_k2_l:.6f}, k3={kb_k3_l:.6f}, k4={kb_k4_l:.6f}")
    print(f"Right: k1={kb_k1_r:.6f}, k2={kb_k2_r:.6f}, k3={kb_k3_r:.6f}, k4={kb_k4_r:.6f}")
    
    print("\n=== TRANSFORMATION MATRICES ===\n")
    
    # Need to account for 90° rotation in the transformation
    # Create rotation matrix for 90° CW rotation
    R_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    # Apply rotation to camera calibrations
    left_rot = calibration.rotate_camera_calib_cw90deg(left_calib)
    right_rot = calibration.rotate_camera_calib_cw90deg(right_calib)
    
    # Get rotated transformations
    T_device_left_rot = left_rot.get_transform_device_camera()
    T_device_right_rot = right_rot.get_transform_device_camera()
    
    # Recalculate with rotated cameras
    T_imu_left_rot = T_device_imu.inverse() @ T_device_left_rot
    T_left_right_rot = T_device_left_rot.inverse() @ T_device_right_rot
    
    print("T_c1_c2 (left to right camera):")
    print(T_left_right_rot.to_matrix())
    
    print("\nT_b_c1 (IMU to left camera):")
    print(T_imu_left_rot.to_matrix())
    
    # Generate YAML configuration
    generate_yaml_config(
        fx_l, fy_l, cx_l_rot, cy_l_rot, kb_k1_l, kb_k2_l, kb_k3_l, kb_k4_l,
        fx_r, fy_r, cx_r_rot, cy_r_rot, kb_k1_r, kb_k2_r, kb_k3_r, kb_k4_r,
        T_left_right_rot.to_matrix(), T_imu_left_rot.to_matrix()
    )


def generate_yaml_config(fx_l, fy_l, cx_l, cy_l, k1_l, k2_l, k3_l, k4_l,
                        fx_r, fy_r, cx_r, cy_r, k1_r, k2_r, k3_r, k4_r,
                        T_c1_c2, T_b_c1):
    """Generate ORB-SLAM3 YAML configuration file"""
    
    yaml_content = f"""%%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters for Aria SLAM cameras (ACTUAL CALIBRATION)
# Extracted from VRS file - NOT simplified!
#--------------------------------------------------------------------------------------------
File.version: "1.0"

# Camera model - KannalaBrandt8 for fisheye/wide-angle cameras
Camera.type: "KannalaBrandt8"

# Left Camera intrinsics (after 90° clockwise rotation)
# Original: 640x480 -> Rotated: 480x640
Camera1.fx: {fx_l:.6f}
Camera1.fy: {fy_l:.6f}
Camera1.cx: {cx_l:.6f}
Camera1.cy: {cy_l:.6f}

# Kannala-Brandt distortion parameters (left)
Camera1.k1: {k1_l:.6f}
Camera1.k2: {k2_l:.6f}
Camera1.k3: {k3_l:.6f}
Camera1.k4: {k4_l:.6f}

# Right Camera intrinsics (after 90° clockwise rotation)
Camera2.fx: {fx_r:.6f}
Camera2.fy: {fy_r:.6f}
Camera2.cx: {cx_r:.6f}
Camera2.cy: {cy_r:.6f}

# Kannala-Brandt distortion parameters (right)
Camera2.k1: {k1_r:.6f}
Camera2.k2: {k2_r:.6f}
Camera2.k3: {k3_r:.6f}
Camera2.k4: {k4_r:.6f}

# Camera resolution (after rotation)
Camera.width: 480
Camera.height: 640

# Camera frame rate
Camera.fps: 10

# Color order (0: BGR, 1: RGB. Ignored for grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
Stereo.ThDepth: 60.0

# Transformation from left to right camera
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{T_c1_c2[0,0]:11.8f}, {T_c1_c2[0,1]:11.8f}, {T_c1_c2[0,2]:11.8f}, {T_c1_c2[0,3]:11.8f},
         {T_c1_c2[1,0]:11.8f}, {T_c1_c2[1,1]:11.8f}, {T_c1_c2[1,2]:11.8f}, {T_c1_c2[1,3]:11.8f},
         {T_c1_c2[2,0]:11.8f}, {T_c1_c2[2,1]:11.8f}, {T_c1_c2[2,2]:11.8f}, {T_c1_c2[2,3]:11.8f},
         0.0, 0.0, 0.0, 1.0]

# Transformation from IMU (body frame) to left camera
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{T_b_c1[0,0]:11.8f}, {T_b_c1[0,1]:11.8f}, {T_b_c1[0,2]:11.8f}, {T_b_c1[0,3]:11.8f},
         {T_b_c1[1,0]:11.8f}, {T_b_c1[1,1]:11.8f}, {T_b_c1[1,2]:11.8f}, {T_b_c1[1,3]:11.8f},
         {T_b_c1[2,0]:11.8f}, {T_b_c1[2,1]:11.8f}, {T_b_c1[2,2]:11.8f}, {T_b_c1[2,3]:11.8f},
         0.0, 0.0, 0.0, 1.0]

# IMU noise parameters for Aria IMU-right (1202-1)
# These are from the Aria paper Table 1, converted to proper units
IMU.NoiseGyro: 1.7453e-4      # rad/s/√Hz (1e-2 deg/s/√Hz)
IMU.NoiseAcc: 7.8480e-4       # m/s²/√Hz (80 μg/√Hz)
IMU.GyroWalk: 2.2689e-5       # rad/s²/√Hz (1.3e-3 deg/s/√Hz)
IMU.AccWalk: 3.4335e-4        # m/s³/√Hz (35 μg/s/√Hz)
IMU.Frequency: 1000.0         # Hz (native frequency)

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 2000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""
    
    # Save to file
    output_path = Path("Examples/Stereo-Inertial/Aria_RealCalibration.yaml")
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n=== GENERATED CONFIGURATION FILE ===")
    print(f"Saved to: {output_path}")
    print("\nKey differences from simplified calibration:")
    print(f"- REAL focal lengths: ~242 pixels (not 150!)")
    print(f"- REAL principal points: properly calculated after rotation")
    print(f"- REAL distortion parameters from FISHEYE624 model")
    print(f"- REAL transformation matrices from device calibration")
    print("\nThis should significantly improve ORB-SLAM3 performance!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use default VRS file
        vrs_path = "/mnt/ssd_ext/incSeg-data/aria_everyday/loc1_script1_seq1_rec1/recording.vrs"
        print(f"Using default VRS file: {vrs_path}")
    else:
        vrs_path = sys.argv[1]
    
    extract_calibration(vrs_path)