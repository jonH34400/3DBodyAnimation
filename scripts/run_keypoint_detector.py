import argparse
import os
import cv2
import mediapipe as mp
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Face, Hands, and/or Pose detection on a directory of images."
    )
    parser.add_argument("--input_dir", default="input_images", help="Input directory with images (jpg/png).")
    parser.add_argument("--output_dir", default="output", help="Base output directory (default: output).")

    parser.add_argument("--face", action="store_true", help="Enable face detection.")
    parser.add_argument("--hands", action="store_true", help="Enable hand detection.")
    parser.add_argument("--pose", action="store_true", help="Enable pose detection.")

    return parser.parse_args()


def save_keypoints_csv(keypoints, header, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(keypoints)

def main():
    args = parse_args()

    if not (args.face or args.hands or args.pose):
        raise ValueError("You must specify at least one of --face, --hands, or --pose.")

    output_images_dir = os.path.join(args.output_dir, "images")
    keypoints_dir = os.path.join(args.output_dir, "keypoints")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)

    mp_draw = mp.solutions.drawing_utils
    face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) if args.face else None
    hands_det = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) if args.hands else None
    pose_det = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) if args.pose else None

    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for fn in files:
        img_path = os.path.join(args.input_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {fn}, couldn't read as image.")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        basename = os.path.splitext(fn)[0]

        if args.face:
            face_res = face_det.process(rgb)
            face_keypoints = []
            if face_res.detections:
                for d in face_res.detections:
                    bbox = d.location_data.relative_bounding_box
                    face_keypoints.append([bbox.xmin, bbox.ymin, bbox.width, bbox.height, d.score[0]])
                    mp_draw.draw_detection(img, d)
            if face_keypoints:
                save_keypoints_csv(
                    face_keypoints,
                    ["xmin", "ymin", "width", "height", "score"], 
                    os.path.join(keypoints_dir, f"face_keypoints_{basename}.csv")
                )

        if args.hands:
            hands_res = hands_det.process(rgb)
            hands_keypoints = []
            if hands_res.multi_hand_landmarks:
                for idx, h in enumerate(hands_res.multi_hand_landmarks):
                    for i, lmk in enumerate(h.landmark):
                        hands_keypoints.append([idx, i, lmk.x, lmk.y, lmk.z])
                    mp_draw.draw_landmarks(img, h, mp.solutions.hands.HAND_CONNECTIONS)
            if hands_keypoints:
                save_keypoints_csv(
                    hands_keypoints,
                    ["hand_idx", "lmk_idx", "x", "y", "z"], 
                    os.path.join(keypoints_dir, f"hands_keypoints_{basename}.csv")
                )

        if args.pose:
            pose_res = pose_det.process(rgb)
            pose_keypoints = []
            if pose_res.pose_landmarks:
                for i, lmk in enumerate(pose_res.pose_landmarks.landmark):
                    pose_keypoints.append([i, lmk.x, lmk.y, lmk.z, lmk.visibility])
                mp_draw.draw_landmarks(img, pose_res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            if pose_keypoints:
                save_keypoints_csv(
                    pose_keypoints,
                    ["lmk_idx", "x", "y", "z", "visibility"], 
                    os.path.join(keypoints_dir, f"pose_keypoints_{basename}.csv")
                )

        out_path = os.path.join(output_images_dir, fn)
        cv2.imwrite(out_path, img)
        print(f"Processed {fn}")

    if face_det: face_det.close()
    if hands_det: hands_det.close()
    if pose_det: pose_det.close()
    print("Completed. Annotated images in", output_images_dir, "and keypoints in", keypoints_dir)

if __name__ == "__main__":
    main()
