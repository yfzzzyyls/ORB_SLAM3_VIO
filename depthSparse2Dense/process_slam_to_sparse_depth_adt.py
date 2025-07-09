#!/usr/bin/env python3
"""
Post-process SLAM tracking data to create sparse depth maps for training
Adapted for ADT dataset pipeline
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import json

class SLAMDataProcessorADT:
    def __init__(self, tracking_dir, tumvi_dir, output_dir):
        self.tracking_dir = Path(tracking_dir)
        self.tumvi_dir = Path(tumvi_dir)
        self.output_dir = Path(output_dir)
        
        # ADT SLAM camera parameters (after 90° rotation)
        self.camera_params = {
            'width': 640,
            'height': 480,
            'fx': 242.7,
            'fy': 242.7,
            'cx': 318.08,
            'cy': 235.65
        }
        
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
    
    def create_sparse_depth_map(self, features):
        """Create sparse depth map from features"""
        height = self.camera_params['height']
        width = self.camera_params['width']
        
        sparse_depth = np.zeros((height, width), dtype=np.float32)
        confidence = np.zeros((height, width), dtype=np.float32)
        
        valid_points = 0
        for feat in features:
            if feat['depth'] > 0 and feat['depth'] < 10.0:  # Valid depth
                u, v = int(round(feat['u'])), int(round(feat['v']))
                if 0 <= u < width and 0 <= v < height:
                    # Use maximum depth if multiple features project to same pixel
                    if sparse_depth[v, u] == 0 or feat['depth'] < sparse_depth[v, u]:
                        sparse_depth[v, u] = feat['depth']
                        confidence[v, u] = min(feat['confidence'], 10) / 10.0
                        valid_points += 1
        
        return sparse_depth, confidence, valid_points
    
    def load_timestamp_mapping(self):
        """Load timestamp mapping from TUM-VI conversion"""
        timestamps_file = self.tumvi_dir / 'mav0' / 'timestamps.txt'
        
        timestamp_to_frame = {}
        with open(timestamps_file, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('#'):
                    ts_ns = int(line)
                    ts_s = ts_ns / 1e9
                    timestamp_to_frame[ts_s] = idx
        
        return timestamp_to_frame
    
    def process_sequence(self):
        """Process all tracking files in the sequence"""
        tracking_files = sorted(self.tracking_dir.glob('frame_*.txt'))
        print(f"Found {len(tracking_files)} tracking files")
        
        # Load timestamp mapping
        ts_mapping = self.load_timestamp_mapping()
        
        # Save camera intrinsics
        K = np.array([[self.camera_params['fx'], 0, self.camera_params['cx']],
                      [0, self.camera_params['fy'], self.camera_params['cy']],
                      [0, 0, 1]])
        np.save(self.output_dir / 'metadata' / 'intrinsics.npy', K)
        
        # Save camera params
        with open(self.output_dir / 'metadata' / 'camera_params.json', 'w') as f:
            json.dump(self.camera_params, f, indent=2)
        
        # Process each frame
        frame_data = []
        valid_frames = 0
        
        for tracking_file in tqdm(tracking_files, desc="Processing frames"):
            # Parse tracking data
            data = self.parse_tracking_file(tracking_file)
            
            # Create sparse depth map
            sparse_depth, confidence, num_valid = self.create_sparse_depth_map(data['features'])
            
            # Skip frames with too few points
            if num_valid < 10:
                continue
            
            # Find corresponding RGB image
            rgb_dir = self.tumvi_dir / 'mav0' / 'cam0' / 'data'
            
            # Find closest timestamp
            closest_ts = min(ts_mapping.keys(), key=lambda x: abs(x - data['timestamp']))
            frame_idx = ts_mapping[closest_ts]
            
            # Load RGB image
            rgb_filename = f"{frame_idx:010d}.png"
            rgb_path = rgb_dir / rgb_filename
            
            if rgb_path.exists():
                # Save sparse depth
                frame_str = f"{data['frame_id']:06d}"
                np.save(self.output_dir / 'sparse_depth' / f'{frame_str}.npy', sparse_depth)
                np.save(self.output_dir / 'sparse_depth' / f'{frame_str}_conf.npy', confidence)
                
                # Save pose
                np.save(self.output_dir / 'poses' / f'{frame_str}.npy', data['pose'])
                
                # Copy RGB image
                rgb = cv2.imread(str(rgb_path))
                cv2.imwrite(str(self.output_dir / 'rgb' / f'{frame_str}.png'), rgb)
                
                # Store metadata
                frame_data.append({
                    'frame_id': data['frame_id'],
                    'timestamp': data['timestamp'],
                    'tumvi_frame': frame_idx,
                    'num_features': num_valid,
                    'sparsity': num_valid / (height * width)
                })
                
                valid_frames += 1
        
        # Save frame metadata
        with open(self.output_dir / 'metadata' / 'frames.json', 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {self.output_dir}")
        print(f"Valid frames with sparse depth: {valid_frames}/{len(tracking_files)}")
        print(f"Average sparsity: {np.mean([f['sparsity'] for f in frame_data]):.4%}")
        
    def visualize_sample(self, frame_id=None):
        """Visualize a sample frame"""
        # Get first available frame if not specified
        if frame_id is None:
            sparse_files = sorted(self.output_dir.glob('sparse_depth/[0-9]*.npy'))
            if sparse_files:
                frame_id = int(sparse_files[0].stem)
            else:
                print("No sparse depth files found!")
                return
        
        frame_str = f"{frame_id:06d}"
        
        # Load data
        sparse_depth = np.load(self.output_dir / 'sparse_depth' / f'{frame_str}.npy')
        rgb_path = self.output_dir / 'rgb' / f'{frame_str}.png'
        
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            print(f"RGB image not found: {rgb_path}")
            return
        
        # Create visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image (SLAM Camera)')
        axes[0, 0].axis('off')
        
        # Sparse depth
        sparse_viz = sparse_depth.copy()
        sparse_viz[sparse_viz == 0] = np.nan
        im1 = axes[0, 1].imshow(sparse_viz, cmap='viridis', vmin=0, vmax=5)
        axes[0, 1].set_title(f'Sparse Depth ({np.sum(sparse_depth > 0)} points)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Sparse points overlay
        axes[1, 0].imshow(rgb)
        y, x = np.where(sparse_depth > 0)
        scatter = axes[1, 0].scatter(x, y, c=sparse_depth[y, x], s=5, cmap='viridis', vmin=0, vmax=5)
        axes[1, 0].set_title('Sparse Points on RGB')
        axes[1, 0].axis('off')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Statistics
        axes[1, 1].axis('off')
        stats_text = f"Frame ID: {frame_id}\n"
        stats_text += f"Valid points: {np.sum(sparse_depth > 0)}\n"
        stats_text += f"Sparsity: {np.sum(sparse_depth > 0) / sparse_depth.size:.2%}\n"
        stats_text += f"Depth range: [{sparse_depth[sparse_depth > 0].min():.2f}, {sparse_depth[sparse_depth > 0].max():.2f}] m\n"
        stats_text += f"Mean depth: {sparse_depth[sparse_depth > 0].mean():.2f} m"
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'visualization_frame_{frame_str}.png', dpi=150)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process SLAM tracking data for sparse-to-dense training (ADT)")
    parser.add_argument('tracking_dir', help='Directory with SLAM tracking output files (e.g., tracking_data_*)')
    parser.add_argument('tumvi_dir', help='Directory with TUM-VI format data from aria_to_tumvi.py')
    parser.add_argument('output_dir', help='Output directory for processed sparse depth data')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample output')
    parser.add_argument('--frame-id', type=int, help='Specific frame ID to visualize')
    
    args = parser.parse_args()
    
    processor = SLAMDataProcessorADT(
        args.tracking_dir,
        args.tumvi_dir,
        args.output_dir
    )
    
    processor.process_sequence()
    
    if args.visualize:
        processor.visualize_sample(args.frame_id)


if __name__ == '__main__':
    main()