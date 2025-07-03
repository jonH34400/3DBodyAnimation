import cv2
import os
from natsort import natsorted  # for sorted filenames (frame_0001.png, frame_0002.png, ...)

video_data = "video8"  # Adjust 

# Path to folder with annotated images
FRAMES_DIR = "data/frames_annotated/" + video_data
OUTPUT_VIDEO = "data/videos_annotated/annotated_" + video_data + ".mp4"

FPS = 10  # Frames per second in the video

# Get all PNG files, sorted
frame_files = natsorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")])

# Load first image to determine size
first_frame = cv2.imread(os.path.join(FRAMES_DIR, frame_files[0]))
height, width, _ = first_frame.shape

# Initialize VideoWriter
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Codec for .mp4
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

# Add all frames
for filename in frame_files:
    frame = cv2.imread(os.path.join(FRAMES_DIR, filename))
    video_writer.write(frame)

video_writer.release()
print(f"✅ Video saved at {OUTPUT_VIDEO}")
