#!/usr/bin/env python3
"""
Download complete ADT sequence including depth data.
"""

import os
import json
import requests
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url, filepath):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_sequence(sequence_name, seq_data, output_base, components):
    """Download one sequence."""
    # Create output directory
    output_dir = Path(output_base) / sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {sequence_name} to {output_dir}")
    print(f"Components: {components}")
    print("-" * 60)
    
    # Download each component
    for component in components:
        if component not in seq_data:
            print(f"\nWarning: {component} not found in sequence data")
            continue
        
        comp_data = seq_data[component]
        filename = comp_data['filename']
        url = comp_data['download_url']
        size_mb = comp_data['file_size_bytes'] / (1024 * 1024)
        
        print(f"\n{component}: {filename} ({size_mb:.1f} MB)")
        
        output_path = output_dir / filename
        
        # Skip if already exists with correct size
        if output_path.exists():
            existing_size = output_path.stat().st_size
            if existing_size == comp_data['file_size_bytes']:
                print(f"  Already exists with correct size, skipping...")
                continue
        
        # Download
        download_file(url, output_path)
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(output_path, 'r') as zf:
                zf.extractall(output_dir)
            print(f"  ✓ Extracted successfully")
            
            # List extracted files
            if component == 'depth':
                print("  Extracted files:")
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        if 'depth' in f and f.endswith('.vrs'):
                            size_gb = os.path.getsize(os.path.join(root, f)) / (1024**3)
                            print(f"    - {f} ({size_gb:.2f} GB)")
    
    print(f"\n✓ Download complete: {output_dir}")
    
    # Check what we have
    print("\nSequence contents:")
    for item in sorted(output_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024**2)
            print(f"  {item.name} ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description='Download complete ADT sequences')
    parser.add_argument('--sequences', nargs='+', 
                       help='Sequence names to download (space-separated)')
    parser.add_argument('--sequence', 
                       help='Single sequence name (deprecated, use --sequences)')
    parser.add_argument('--all', action='store_true',
                       help='Download all available sequences')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip sequences that already have a directory')
    parser.add_argument('--output', default='/mnt/ssd_ext/incSeg-data/adt',
                       help='Output directory')
    parser.add_argument('--components', nargs='+', 
                       default=['main_vrs', 'depth', 'main_groundtruth', 'segmentation'],
                       choices=['video_main_rgb', 'main_vrs', 'main_groundtruth', 
                               'segmentation', 'depth', 'synthetic', 'mps_slam_trajectories',
                               'mps_slam_calibration', 'mps_slam_points', 'mps_eye_gaze'],
                       help='Components to download')
    
    args = parser.parse_args()
    
    # Load JSON
    json_file = Path(__file__).parent.parent / 'ADT_download_urls.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Determine which sequences to download
    sequences_to_download = []
    
    if args.all:
        sequences_to_download = list(data['sequences'].keys())
    elif args.sequences:
        sequences_to_download = args.sequences
    elif args.sequence:
        sequences_to_download = [args.sequence]
    else:
        print("Error: Must specify --sequences, --sequence, or --all")
        print(f"\nAvailable sequences:")
        for seq in data['sequences'].keys():
            print(f"  - {seq}")
        return
    
    # Filter out non-existent sequences
    valid_sequences = []
    for seq in sequences_to_download:
        if seq not in data['sequences']:
            print(f"Warning: Sequence '{seq}' not found, skipping...")
        else:
            valid_sequences.append(seq)
    
    if not valid_sequences:
        print("No valid sequences to download!")
        return
    
    # Check for existing sequences if skip flag is set
    if args.skip_existing:
        remaining = []
        for seq in valid_sequences:
            seq_dir = Path(args.output) / seq
            if seq_dir.exists():
                print(f"Skipping existing sequence: {seq}")
            else:
                remaining.append(seq)
        valid_sequences = remaining
    
    print(f"\nWill download {len(valid_sequences)} sequences:")
    for seq in valid_sequences:
        print(f"  - {seq}")
    
    # Download each sequence
    for i, sequence_name in enumerate(valid_sequences):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(valid_sequences)}] Processing {sequence_name}")
        print(f"{'='*60}")
        
        seq_data = data['sequences'][sequence_name]
        download_sequence(sequence_name, seq_data, args.output, args.components)

if __name__ == '__main__':
    main()