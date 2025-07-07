#!/usr/bin/env python3
"""
Post-process SLAM tracking data to create sparse depth maps for training
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import os

class SLAMDataProcessor:
    def __init__(self, tracking_dir, rgb_dir, output_dir, camera_params):
        self.tracking_dir = Path(tracking_dir)
        self.rgb_dir = Path(rgb_dir)
        self.output_dir = Path(output_dir)
        self.camera_params = camera_params
        
        # Create output directories
        (self.output_dir / 'sparse_depth').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'rgb').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'poses').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
    def parse_tracking_file(self, tracking_file):
        """Parse a single tracking data file"""
        with open(tracking_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        frame_info = lines[1].strip().split()
        frame_id = int(frame_info[0])
        timestamp = float(frame_info[1])
        
        # Parse pose (4x4 matrix)
        pose = np.zeros((4, 4))
        for i in range(4):
            pose[i] = list(map(float, lines[3+i].strip().split()))
        
        # Parse number of features
        num_features = int(lines[8].strip())
        
        # Parse features
        features = []
        for i in range(num_features):
            parts = lines[10+i].strip().split()
            feature = {
                'u': float(parts[0]),
                'v': float(parts[1]),
                'depth': float(parts[2]),
                'point_id': int(parts[3]),
                'confidence': int(parts[4])
            }
            features.append(feature)
        
        return {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'pose': pose,
            'features': features
        }
    
    def create_sparse_depth_map(self, features, height, width):
        """Create sparse depth map from features"""
        sparse_depth = np.zeros((height, width), dtype=np.float32)
        confidence = np.zeros((height, width), dtype=np.float32)
        
        for feat in features:
            if feat['depth'] > 0 and feat['depth'] < 10.0:  # Valid depth
                u, v = int(round(feat['u'])), int(round(feat['v']))
                if 0 <= u < width and 0 <= v < height:
                    # Use maximum depth if multiple features project to same pixel
                    if sparse_depth[v, u] == 0 or feat['depth'] < sparse_depth[v, u]:
                        sparse_depth[v, u] = feat['depth']
                        confidence[v, u] = min(feat['confidence'], 10) / 10.0
        
        return sparse_depth, confidence
    
    def process_sequence(self):
        """Process all tracking files in the sequence"""
        tracking_files = sorted(self.tracking_dir.glob('frame_*.txt'))
        print(f"Found {len(tracking_files)} tracking files")
        
        # Save camera intrinsics
        K = np.array([[self.camera_params['fx'], 0, self.camera_params['cx']],
                      [0, self.camera_params['fy'], self.camera_params['cy']],
                      [0, 0, 1]])
        np.save(self.output_dir / 'metadata' / 'intrinsics.npy', K)
        
        # Process each frame
        frame_data = []
        for tracking_file in tqdm(tracking_files, desc="Processing frames"):
            # Parse tracking data
            data = self.parse_tracking_file(tracking_file)
            
            # Create sparse depth map
            sparse_depth, confidence = self.create_sparse_depth_map(
                data['features'], 
                self.camera_params['height'],
                self.camera_params['width']
            )
            
            # Save sparse depth
            frame_str = f"{data['frame_id']:06d}"
            np.save(self.output_dir / 'sparse_depth' / f'{frame_str}.npy', sparse_depth)
            np.save(self.output_dir / 'sparse_depth' / f'{frame_str}_conf.npy', confidence)
            
            # Save pose
            np.save(self.output_dir / 'poses' / f'{frame_str}.npy', data['pose'])
            
            # Copy corresponding RGB image if available
            rgb_pattern = f"*{data['timestamp']:.6f}*"
            rgb_files = list(self.rgb_dir.glob(rgb_pattern))
            if not rgb_files:
                # Try with frame ID
                rgb_files = list(self.rgb_dir.glob(f"*{frame_str}*"))
            
            if rgb_files:
                rgb = cv2.imread(str(rgb_files[0]))
                rgb = cv2.resize(rgb, (self.camera_params['width'], self.camera_params['height']))
                cv2.imwrite(str(self.output_dir / 'rgb' / f'{frame_str}.png'), rgb)
            
            # Store metadata
            frame_data.append({
                'frame_id': data['frame_id'],
                'timestamp': data['timestamp'],
                'num_features': len([f for f in data['features'] if f['depth'] > 0]),
                'has_rgb': len(rgb_files) > 0
            })
        
        # Save frame metadata
        import json
        with open(self.output_dir / 'metadata' / 'frames.json', 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {self.output_dir}")
        print(f"Total frames with sparse depth: {len(frame_data)}")
        
    def visualize_sample(self, frame_id=0):
        """Visualize a sample frame"""
        frame_str = f"{frame_id:06d}"
        
        # Load data
        sparse_depth = np.load(self.output_dir / 'sparse_depth' / f'{frame_str}.npy')
        rgb_path = self.output_dir / 'rgb' / f'{frame_str}.png'
        
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((self.camera_params['height'], self.camera_params['width'], 3))
        
        # Create visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # Sparse depth
        sparse_viz = sparse_depth.copy()
        sparse_viz[sparse_viz == 0] = np.nan
        im1 = axes[1].imshow(sparse_viz, cmap='viridis')
        axes[1].set_title(f'Sparse Depth ({np.sum(sparse_depth > 0)} points)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Sparse points overlay
        axes[2].imshow(rgb)
        y, x = np.where(sparse_depth > 0)
        axes[2].scatter(x, y, c=sparse_depth[y, x], s=1, cmap='viridis')
        axes[2].set_title('Sparse Points on RGB')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_visualization.png')
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracking_dir', help='Directory with SLAM tracking output files')
    parser.add_argument('rgb_dir', help='Directory with RGB images from TUM-VI format')
    parser.add_argument('output_dir', help='Output directory for processed data')
    
    # Camera parameters (Aria SLAM camera after rotation)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--fx', type=float, default=242.7)
    parser.add_argument('--fy', type=float, default=242.7)
    parser.add_argument('--cx', type=float, default=235.65)
    parser.add_argument('--cy', type=float, default=318.08)
    
    parser.add_argument('--visualize', action='store_true', help='Visualize sample output')
    
    args = parser.parse_args()
    
    camera_params = {
        'width': args.width,
        'height': args.height,
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    processor = SLAMDataProcessor(
        args.tracking_dir,
        args.rgb_dir,
        args.output_dir,
        camera_params
    )
    
    processor.process_sequence()
    
    if args.visualize:
        processor.visualize_sample()


if __name__ == '__main__':
    main()