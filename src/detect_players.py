from ultralytics import YOLO
import cv2
import os

def detect_and_crop(video_path, model_path, save_dir):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]
            filename = f"{video_name}_frame{frame_id}_person{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), crop)

        frame_id += 1
    cap.release()