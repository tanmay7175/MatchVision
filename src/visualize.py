import cv2
import os

def draw_ids(video_path, match_dict, save_path):
    cap = cv2.VideoCapture(video_path)
    out = None
    frame_id = 0
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw only if frame ID appears in file name
        for (file1, file2), pid in match_dict.items():
            if f"frame{frame_id}_" in file1:
                y_offset = 50 + int(pid[1:]) * 20
                cv2.putText(frame, f"ID: {pid}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(save_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
