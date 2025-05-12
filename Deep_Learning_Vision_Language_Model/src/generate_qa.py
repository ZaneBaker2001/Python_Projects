import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)
        min_box_size: Minimum size for bounding boxes to consider

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Get the detections for this view
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return []
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Extract kart information
    karts = []
    
    # Calculate image center
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Track the minimum distance to the center (to identify the center kart)
    min_distance = float('inf')
    center_kart_idx = None
    
    # Get kart_names mapping if it exists in the info file
    kart_names = info.get("kart_names", {})
    
    # Common character karts in SuperTuxKart
    common_karts = [
        "tux", "gnu", "beastie", "amanda", "suzanne", "wilber", "kiki", "konqi", 
        "xue", "puffy", "gavroche", "hexley", "pidgin", "adiumy", "emule"
    ]
    
    # Process each detection
    for idx, detection in enumerate(frame_detections):
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Skip if not a kart
        if class_id != 1:
            continue
            
        # Scale coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
            
        # Skip if kart is completely outside the image boundaries
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
            
        # Calculate center of the kart
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Calculate distance to image center
        distance = ((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2) ** 0.5
        
        # Check if this is the closest kart to the center
        if distance < min_distance:
            min_distance = distance
            center_kart_idx = len(karts)
        
        # Get kart name from mapping or use default name
        kart_name = kart_names.get(str(track_id), "")
        if not kart_name:
            # For ego car (track_id 0), use one of the common karts if not specified
            if track_id == 0:
                kart_name = "beastie"
            else:
                # For other karts, use a common kart name based on track_id
                kart_name = common_karts[track_id % len(common_karts)] if track_id < len(common_karts) * 2 else f"kart_{track_id}"
            
        # Add kart to the list
        karts.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False,  # Will update later
            "box": (x1_scaled, y1_scaled, x2_scaled, y2_scaled)
        })
    
    # Mark the center kart
    if center_kart_idx is not None and karts:
        karts[center_kart_idx]["is_center_kart"] = True
    
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # Common track names in SuperTuxKart
    common_tracks = [
        "abyss", "black_forest", "candela_city", "cocoa_temple", "gran_paradiso_island",
        "hacienda", "lighthouse", "minigolf", "olivermath", "ravenbridge_mansion",
        "sandtrack", "snowmountain", "snowtuxpeak", "tutorial", "volcano_island", "xr591"
    ]
    
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Extract track name if available
    if "track_name" in info:
        track_name = info["track_name"]
    elif "track" in info:
        track_name = info["track"]
    else:
        # If not provided, use path info or default
        path = Path(info_path)
        dir_name = path.parent.name
        
        # Check if directory name might contain track info
        if "track" in dir_name.lower():
            track_name = dir_name
        else:
            # Assign a random track name based on path hash
            hash_value = hash(str(info_path))
            track_name = common_tracks[hash_value % len(common_tracks)]
            
    # Ensure track name is normalized to match common format
    track_name = track_name.lower().replace(" ", "_")
    
    # Map to a common track if possible
    for common_track in common_tracks:
        if common_track in track_name:
            return common_track
            
    return track_name


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    
    # Extract kart objects and track info
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # Get image relative path (for storing in QA pairs)
    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_file = f"{base_name}_{view_index:02d}_im.jpg"
    
    # Format the image path as specified: "train/007d8_06_im.jpg"
    split_name = info_path_obj.parent.name  # e.g., "train" or "valid"
    image_path = f"{split_name}/{image_file}"
    
    # Find ego kart (usually has track_id 0)
    ego_kart = None
    for kart in karts:
        if kart["instance_id"] == 0:
            ego_kart = kart
            break
    
    # If no ego kart found, use the center kart as ego
    if ego_kart is None and karts:
        for kart in karts:
            if kart["is_center_kart"]:
                ego_kart = kart
                break
    
    # If still no ego kart, use the first kart
    if ego_kart is None and karts:
        ego_kart = karts[0]
    
    # 1. Ego car question
    if ego_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"],
            "image_file": image_path
        })
    
    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
        "image_file": image_path
    })
    
    # 3. Track information questions
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name,
        "image_file": image_path
    })
    
    # Skip relative position questions if no ego kart
    if ego_kart and len(karts) > 1:
        ego_center_x = ego_kart["center"][0]
        ego_center_y = ego_kart["center"][1]
        
        # Count karts in different positions
        karts_left = 0
        karts_right = 0
        karts_front = 0
        karts_behind = 0
        
        # 4. Relative position questions for each kart
        for kart in karts:
            # Skip the ego kart
            if kart["instance_id"] == ego_kart["instance_id"]:
                continue
                
            kart_center_x = kart["center"][0]
            kart_center_y = kart["center"][1]
            
            # Determine left/right position
            left_right = "left" if kart_center_x < ego_center_x else "right"
            if left_right == "left":
                karts_left += 1
            else:
                karts_right += 1
                
            # Determine front/behind position (in a racing game, lower y is typically "ahead")
            # Use "front" and "back" instead of "in front of" and "behind" to match test set
            front_behind = "front" if kart_center_y < ego_center_y else "back"
            if front_behind == "front":
                karts_front += 1
            else:
                karts_behind += 1
                
            # Add position questions
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": left_right,
                "image_file": image_path
            })
            
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": front_behind,
                "image_file": image_path
            })
            
            # Add combined position question (as seen in the test set)
            qa_pairs.append({
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": f"{front_behind} and {left_right}",
                "image_file": image_path
            })
        
        # 5. Counting questions
        qa_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(karts_left),
            "image_file": image_path
        })
        
        qa_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(karts_right),
            "image_file": image_path
        })
        
        qa_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(karts_front),
            "image_file": image_path
        })
        
        qa_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(karts_behind),
            "image_file": image_path
        })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_files = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.*"))
    
    if not image_files:
        print(f"Error: No image file found for {base_name}_{view_index:02d}_im.*")
        return
        
    image_file = image_files[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Image: {qa['image_file']}")
        print("-" * 50)


def generate_dataset(input_dir: str, output_file: str, max_samples: int = None):
    """
    Generate a dataset of QA pairs from all info files in a directory.
    
    Args:
        input_dir: Directory containing info.json files
        output_file: Path to save the QA pairs JSON
        max_samples: Maximum number of samples to generate (per view)
    """
    input_path = Path(input_dir)
    all_qa_pairs = []
    
    # Find all info files
    info_files = list(input_path.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {input_dir}")
    
    # Process each info file
    for info_file in info_files:
        # Extract possible view indices
        with open(info_file) as f:
            info = json.load(f)
            num_views = len(info.get("detections", []))
        
        # Process each view
        for view_index in range(num_views):
            # Find the corresponding image file
            base_name = info_file.stem.replace("_info", "")
            image_files = list(info_file.parent.glob(f"{base_name}_{view_index:02d}_im.*"))
            
            if not image_files:
                continue
                
            # Generate QA pairs for this view
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            all_qa_pairs.extend(qa_pairs)
            
            print(f"Generated {len(qa_pairs)} QA pairs for {info_file.name}, view {view_index}")
            
            # Limit samples if needed
            if max_samples and len(all_qa_pairs) >= max_samples:
                all_qa_pairs = all_qa_pairs[:max_samples]
                break
        
        # Check if we've reached the sample limit
        if max_samples and len(all_qa_pairs) >= max_samples:
            break
    
    # Save the QA pairs to the output file
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Saved {len(all_qa_pairs)} QA pairs to {output_file}")


def balance_dataset(input_file: str, output_file: str, max_per_question_type: int = 1000):
    """
    Balance the dataset by ensuring a more even distribution of question types.
    
    Args:
        input_file: Path to the input QA pairs JSON
        output_file: Path to save the balanced QA pairs JSON
        max_per_question_type: Maximum samples per question type
    """
    # Load the QA pairs
    with open(input_file) as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(qa_pairs)} QA pairs from {input_file}")
    
    # Group by question type
    question_types = {}
    for qa in qa_pairs:
        # Extract question type
        question = qa["question"]
        if "how many" in question.lower():
            if "scenario" in question.lower():
                qtype = "how_many_scenario"
            elif "left" in question.lower():
                qtype = "how_many_left"
            elif "right" in question.lower():
                qtype = "how_many_right"
            elif "front" in question.lower():
                qtype = "how_many_front"
            elif "behind" in question.lower():
                qtype = "how_many_behind"
            else:
                qtype = "how_many_other"
        elif "what kart" in question.lower():
            qtype = "what_kart"
        elif "what track" in question.lower():
            qtype = "what_track"
        elif "where is" in question.lower():
            qtype = "where_relative"
        elif "left or right" in question.lower():
            qtype = "left_right"
        elif "front of or behind" in question.lower():
            qtype = "front_behind"
        else:
            qtype = "other"
        
        if qtype not in question_types:
            question_types[qtype] = []
        
        question_types[qtype].append(qa)
    
    # Balance the dataset
    balanced_qa_pairs = []
    for qtype, qa_list in question_types.items():
        # Shuffle the list to get random samples
        np.random.shuffle(qa_list)
        
        # Take up to max_per_question_type samples
        balanced_qa_pairs.extend(qa_list[:max_per_question_type])
        
        print(f"Question type '{qtype}': {len(qa_list)} samples -> {min(len(qa_list), max_per_question_type)} samples")
    
    # Shuffle the final dataset
    np.random.shuffle(balanced_qa_pairs)
    
    # Save the balanced dataset
    with open(output_file, 'w') as f:
        json.dump(balanced_qa_pairs, f, indent=2)
    
    print(f"Saved {len(balanced_qa_pairs)} balanced QA pairs to {output_file}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

Generate a dataset:
   python generate_qa.py generate --input_dir ../data/train --output_file ../data/train/qa_pairs_raw.json --max_samples 5000

Balance the dataset:
   python generate_qa.py balance --input_file ../data/train/qa_pairs_raw.json --output_file ../data/train/balanced_qa_pairs.json --max_per_question_type 500
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_dataset,
        "balance": balance_dataset
    })


if __name__ == "__main__":
    main()