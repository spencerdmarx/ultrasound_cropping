import cv2
import numpy as np
from typing import List, Tuple
from remove_background import remove_background
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil

def visualize_crop_boundaries(video_path: str, output_path: str, frame_skip: int = 10):
    """
    Create a visualization of where the crop boundaries would be applied.
    Saves a new video with the crop rectangle drawn on each frame.
    
    Args:
        video_path: path to the input video
        output_path: path where the visualization video will be saved
        frame_skip: process every nth frame for determining crop coordinates
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # First pass: collect crop coordinates
    crop_coords_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Get cropped frame and bounding box from remove_background
            cropped_frame, bbox = remove_background(frame)
            if bbox is not None:
                # Use the actual coordinates from the bounding box
                x1, y1 = bbox.start_x, bbox.start_y
                x2, y2 = bbox.end_x, bbox.end_y
                crop_coords_list.append((x1, y1, x2, y2))
            
        frame_count += 1
    
    # Get median crop coordinates based on area
    if not crop_coords_list:
        raise ValueError(f"No frames could be processed in video: {video_path}")
        
    areas = [(x2-x1) * (y2-y1) for x1, y1, x2, y2 in crop_coords_list]
    median_idx = np.argsort(areas)[len(areas)//2]
    x1, y1, x2, y2 = crop_coords_list[median_idx]
    
    # Second pass: draw rectangles on frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw rectangle showing crop boundaries
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text showing dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        out.write(frame)
    
    cap.release()
    out.release()

def process_video(input_path: str, output_path: str, frame_skip: int = 10, crop_extra_off_top: bool = True):
    """
    Process a video by cropping it based on content and applying histogram equalization.
    
    Args:
        input_path: path to the input video
        output_path: path where the processed video will be saved
        frame_skip: process every nth frame for determining crop coordinates
        crop_extra_off_top: whether to crop 10% from top edge (default: True)
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # First pass: collect crop coordinates
    crop_coords_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Get cropped frame and bounding box from remove_background
            cropped_frame, bbox = remove_background(frame)
            if bbox is not None:
                # Use the actual coordinates from the bounding box
                x1, y1 = bbox.start_x, bbox.start_y
                x2, y2 = bbox.end_x, bbox.end_y
                crop_coords_list.append((x1, y1, x2, y2))
            
        frame_count += 1
    
    # Get median crop coordinates based on area
    if not crop_coords_list:
        raise ValueError(f"No frames could be processed in video: {input_path}")
        
    areas = [(x2-x1) * (y2-y1) for x1, y1, x2, y2 in crop_coords_list]
    median_idx = np.argsort(areas)[len(areas)//2]
    x1, y1, x2, y2 = crop_coords_list[median_idx]
    
    # Create video writer with the crop dimensions. Reduce height by 10% if crop_extra_off_top is True
    if crop_extra_off_top:
        y1 += int((y2 - y1) * 0.1)

    crop_width = x2 - x1
    crop_height = y2 - y1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
    
    # Second pass: crop frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        cropped = frame[y1:y2, x1:x2]
        
        # Write the processed frame
        out.write(cropped)
    
    cap.release()
    out.release()

def save_video_dimensions(input_path: str, output_path: str, dimensions_file: str):
    """
    Save the dimensions of input and output videos to a CSV file.
    
    Args:
        input_path: path to the input video
        output_path: path to the processed video
        dimensions_file: path to the CSV file where dimensions will be saved
    """
    # Get input dimensions
    cap_in = cv2.VideoCapture(input_path)
    in_width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_in.release()
    
    # Get output dimensions
    cap_out = cv2.VideoCapture(output_path)
    out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out.release()
    
    # Create or append to dimensions file
    header = "input_file,input_dimensions,output_dimensions\n"
    if not os.path.exists(dimensions_file):
        with open(dimensions_file, 'w') as f:
            f.write(header)
    
    with open(dimensions_file, 'a') as f:
        f.write(f"{input_path},{in_width}x{in_height},{out_width}x{out_height}\n")

def process_single_video(input_file: Path, output_file: Path, dimensions_file: Path, frame_skip: int, crop_extra_off_top: bool):
    """
    Helper function to process a single video file.
    
    Args:
        input_file: path to input video
        output_file: path to save processed video
        dimensions_file: path to save dimensions
        frame_skip: process every nth frame
        crop_extra_off_top: whether to crop extra from top
    """
    print(f"Processing: {input_file}")
    try:
        process_video(str(input_file), str(output_file), frame_skip, crop_extra_off_top)
        # Use a lock when writing to the shared dimensions file
        with threading.Lock():
            save_video_dimensions(str(input_file), str(output_file), str(dimensions_file))
        print(f"Saved to: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def process_video_directory(input_dir: str, output_dir: str, frame_skip: int = 10, crop_extra_off_top: bool = True, max_workers: int = 4):
    """
    Process all MP4 files in a directory and its subdirectories using multiple threads.
    
    Args:
        input_dir: path to the input directory containing MP4 files
        output_dir: path where the processed videos will be saved
        frame_skip: process every nth frame for determining crop coordinates
        crop_extra_off_top: whether to crop 10% from top edge (default: True)
        max_workers: maximum number of concurrent threads (default: 4)
    """
    # Convert to Path objects for easier manipulation
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dimensions file in the output directory
    dimensions_file = output_path / "video_dimensions.csv"
    
    # Create a list to store all video processing tasks
    video_tasks = []
    
    # Collect all video processing tasks
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                rel_path = Path(root).relative_to(input_path)
                input_file = Path(root) / file
                output_subdir = output_path / rel_path
                output_file = output_subdir / file
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                video_tasks.append((input_file, output_file, dimensions_file, frame_skip, crop_extra_off_top))
    
    # Process videos using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda x: process_single_video(*x), video_tasks)

def analyze_dimension_changes(dimensions_file: str, top_n: int = 50) -> None:
    """
    Analyze the video_dimensions.csv file and print out videos ranked by their area reduction percentage.
    
    Args:
        dimensions_file: path to the CSV file containing video dimensions
        top_n: number of top videos to display (default: 10)
    """
    # Check if file exists
    if not os.path.exists(dimensions_file):
        raise FileNotFoundError(f"Dimensions file not found: {dimensions_file}")
    
    # Store video information
    video_stats = []
    
    # Skip header row and process each line
    with open(dimensions_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            input_file, input_dim, output_dim = line.strip().split(',')
            
            # Parse dimensions
            in_w, in_h = map(int, input_dim.split('x'))
            out_w, out_h = map(int, output_dim.split('x'))
            
            # Calculate areas
            input_area = in_w * in_h
            output_area = out_w * out_h
            
            # Calculate reduction percentage
            reduction_percent = ((input_area - output_area) / input_area) * 100
            
            video_stats.append({
                'file': input_file,
                'input_dims': f"{in_w}x{in_h}",
                'output_dims': f"{out_w}x{out_h}",
                'reduction': reduction_percent
            })
    
    # Sort by reduction percentage (highest first)
    video_stats.sort(key=lambda x: x['reduction'], reverse=True)
    
    # Print results
    print("\nTop videos by area reduction percentage:")
    print("-" * 80)
    print(f"{'Video File':<50} {'Input Dims':<12} {'Output Dims':<12} {'Reduction %':>10}")
    print("-" * 80)
    
    for stat in video_stats[:top_n]:
        print(f"{os.path.basename(stat['file']):<50} "
              f"{stat['input_dims']:<12} "
              f"{stat['output_dims']:<12} "
              f"{stat['reduction']:>9.1f}%")

def copy_videos_with_low_reduction(dimensions_file: str, source_dir: str, target_dir: str, 
                                  threshold: float = 86.0) -> None:
    """
    Copy videos that have a reduction percentage less than or equal to the threshold.
    
    Args:
        dimensions_file: path to the CSV file containing video dimensions
        source_dir: directory containing the processed videos
        target_dir: directory where to copy the filtered videos
        threshold: maximum reduction percentage to include (default: 86.0)
    """
    import shutil
    
    # Check if file exists
    if not os.path.exists(dimensions_file):
        raise FileNotFoundError(f"Dimensions file not found: {dimensions_file}")
    
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Store videos to copy
    videos_to_copy = []
    
    # Skip header row and process each line
    with open(dimensions_file, 'r') as f:
        lines = f.readlines()
        
    # Skip the header row if it exists
    if lines and "input_file" in lines[0].lower():
        lines = lines[1:]
    
    # Process each line
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue  # Skip malformed lines
            
        input_file, input_dim, output_dim = parts
        
        try:
            # Parse dimensions
            in_w, in_h = map(int, input_dim.split('x'))
            out_w, out_h = map(int, output_dim.split('x'))
            
            # Calculate areas
            input_area = in_w * in_h
            output_area = out_w * out_h
            
            # Calculate reduction percentage
            reduction_percent = ((input_area - output_area) / input_area) * 100
            
            # Check if reduction is below threshold
            if reduction_percent <= threshold:
                videos_to_copy.append(input_file)
        except ValueError as e:
            print(f"Error parsing line: {line.strip()} - {str(e)}")
    
    # Copy the videos
    copied_count = 0
    for video_path in videos_to_copy:
        try:
            # Get the category and filename
            # Path format: /path/to/video_nocrop/category/filename.mp4
            parts = video_path.split('/')
            category = parts[-2]  # Get the category (e.g., 'lung')
            filename = parts[-1]  # Get just the filename
            
            # Create category subdirectory in target if it doesn't exist
            category_dir = target_path / category
            category_dir.mkdir(exist_ok=True)
            
            # Source is the processed video with preserved directory structure
            source_file = Path(source_dir) / category / filename
            
            # Target is in the target directory with category subdirectory
            target_file = category_dir / filename
            
            # Copy the file
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                copied_count += 1
                if copied_count % 10 == 0:  # Print progress every 10 files
                    print(f"Copied {copied_count} files so far...")
            else:
                print(f"Source file not found: {source_file}")
        except Exception as e:
            print(f"Error copying {video_path}: {str(e)}")
    
    print(f"\nCopied {copied_count} videos with reduction percentage ≤ {threshold}% to {target_dir}")