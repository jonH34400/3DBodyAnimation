import os
import json
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

video_data = "video8"  # adjust as needed

VIDEO_PATH = "data/videos/" + video_data + ".mp4"   
FRAMES_DIR = "data/frames/" + video_data
AN_FRAMES_DIR = "data/frames_annotated/" + video_data
KEYPOINTS_DIR = "data/keypoints/" + video_data
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(AN_FRAMES_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

pose = mp_pose.Pose(static_image_mode=True)


frame_id = 0
frame_skip = 10 # every 10th frame

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    # save frames
    frame_filename = os.path.join(FRAMES_DIR, f"frame_{frame_id:04d}.png")
    cv2.imwrite(frame_filename, frame)

    # pose estimation
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    keypoints = []
    pose_landmarks = getattr(results, 'pose_landmarks', None)
    if pose_landmarks:
        for lm in pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

    # Visualize: Draw keypoints on image (for debug/visualization)
    annotated_image = image_rgb.copy()
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            pose_landmarks, 
            list(mp_pose.POSE_CONNECTIONS)
        )

    # Convert back to BGR for OpenCV display/saving
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Save annotated image
    annotated_path = os.path.join(AN_FRAMES_DIR, f"frame_{frame_id:04d}_annotated.png")
    cv2.imwrite(annotated_path, annotated_bgr)

    # Save keypoints as JSON
    keypoints_filename = os.path.join(KEYPOINTS_DIR, f"frame_{frame_id:04d}.json")
    with open(keypoints_filename, 'w') as f:
        json.dump(keypoints, f, indent=2)

    frame_id += 1

cap.release()
