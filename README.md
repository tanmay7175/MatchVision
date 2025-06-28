Cross-camera player re-identification using YOLO + ResNet + cosine similarity.

## Directory Structure
```
MatchVision/
├── models/               # best.pt YOLO model
├── videos/               # tacticam.mp4 and broadcast.mp4
├── outputs/
│   ├── crops/
│   └── annotated/
├── features/             # Stores feature .npy files
├── src/
│   ├── detect_players.py
│   ├── extract_features.py
│   ├── match_players.py
│   ├── visualize.py
│   └── evaluate.py
├── main.py
├── streamlit_app.py
├── requirements.txt
└── README.md