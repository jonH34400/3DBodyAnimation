import os
import json
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import yt_dlp
# Suppress OpenCV warnings in headless environments
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


def setup_directories(video_data):
    """
    Create necessary directories for frames, annotations, and keypoints.
    
    Args:
        video_data (str): The video identifier used for directory naming
        
    Returns:
        tuple: Paths to frames, annotated frames, and keypoints directories
    """
    frames_dir = os.path.join("data/frames", video_data)
    an_frames_dir = os.path.join("data/frames_annotated", video_data)
    keypoints_dir = os.path.join("data/keypoints", video_data)
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(an_frames_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    
    return frames_dir, an_frames_dir, keypoints_dir


def extract_keypoints(pose_landmarks):
    """
    Extract keypoints from MediaPipe pose landmarks.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks object
        
    Returns:
        list: List of keypoint dictionaries with x, y, z coordinates and visibility
    """
    keypoints = []
    if pose_landmarks:
        for lm in pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
    return keypoints


def create_annotated_image(image_rgb, pose_landmarks):
    """
    Create an annotated image with pose landmarks drawn.
    
    Args:
        image_rgb (numpy.ndarray): RGB image array
        pose_landmarks: MediaPipe pose landmarks object
        
    Returns:
        numpy.ndarray: Annotated image in BGR format
    """
    annotated_image = image_rgb.copy()
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            pose_landmarks, 
            list(mp_pose.POSE_CONNECTIONS)
        )
    
    # Convert back to BGR for OpenCV saving
    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


def save_frame_data(frame, keypoints, frame_id, frames_dir, an_frames_dir, keypoints_dir, annotated_bgr):
    """
    Save frame, keypoints, and annotated image to respective directories.
    
    Args:
        frame (numpy.ndarray): Original frame
        keypoints (list): List of keypoint dictionaries
        frame_id (int): Frame identifier
        frames_dir (str): Directory for original frames
        an_frames_dir (str): Directory for annotated frames
        keypoints_dir (str): Directory for keypoints JSON files
        annotated_bgr (numpy.ndarray): Annotated frame in BGR format
    """
    # Save original frame
    frame_filename = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
    cv2.imwrite(frame_filename, frame)
    
    # Save annotated frame
    annotated_path = os.path.join(an_frames_dir, f"frame_{frame_id:04d}_annotated.png")
    cv2.imwrite(annotated_path, annotated_bgr)
    
    # Save keypoints as JSON
    keypoints_filename = os.path.join(keypoints_dir, f"frame_{frame_id:04d}.json")
    with open(keypoints_filename, 'w') as f:
        json.dump(keypoints, f, indent=2)


def process_video(video_data="video8"):
    """
    Process the first 50 frames of a video to extract pose keypoints and create annotated frames.
    
    Args:
        video_data (str): Video identifier for file naming
    """
    # Setup paths and directories
    video_path = os.path.join(video_data)
    frames_dir, an_frames_dir, keypoints_dir = setup_directories(os.path.basename(video_data))

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_id = 0
    processed_frames = 0

    try:
        while processed_frames < 1000:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Convert frame to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            pose_landmarks = getattr(results, 'pose_landmarks', None)
            keypoints = extract_keypoints(pose_landmarks)
            annotated_bgr = create_annotated_image(image_rgb, pose_landmarks)

            save_frame_data(frame, keypoints, frame_id, frames_dir, an_frames_dir, keypoints_dir, annotated_bgr)

            frame_id += 1
            processed_frames += 1
    finally:
        cap.release()
        pose.close()

    print(f"Processed {processed_frames} frames (first 50) from {video_data}")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Annotated: {an_frames_dir}")
    print(f"  - Keypoints: {keypoints_dir}")



def download_video(url, output_dir="data/videos"):
    ydl_opts = {
        'format': 'bv[ext=mp4][vcodec^=avc1]+ba/b[ext=mp4][vcodec^=avc1]',
        'merge_output_format': 'mp4',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if ydl_opts.get('merge_output_format'):
            # Update extension if postprocessing changes it (e.g., to mp4)
            filename = os.path.splitext(filename)[0] + f".{ydl_opts['merge_output_format']}"
        return filename

if __name__ == "__main__":
    video_url = "https://www.youtube.com/shorts/VRARSix8A-U"  # ← Replace with your exact URL
    video_filename = download_video(video_url)

    try:
        process_video(video_filename)
    except Exception as e:
        print(f"Error processing video: {e}")
        exit(1)
