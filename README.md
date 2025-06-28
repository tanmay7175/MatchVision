MatchVision: Cross-Camera Player Re-Identification
MatchVision is an end-to-end system for identifying and matching football players captured from different camera angles using deep learning techniques. It combines YOLOv5 for player detection, ResNet for feature extraction, and cosine similarity for re-identification.

Key Features
Player Detection: Uses a YOLOv5 model trained to detect players in match footage.

Feature Extraction: ResNet-based embeddings for each detected player crop.

Cross-View Matching: Cosine similarity is used to match players from two different video sources.

Annotation and Visualization: Player IDs are annotated on the video frames.

Evaluation: Compares predicted matches against ground truth using accuracy score.

Directory Structure
graphql
Copy
Edit
MatchVision/
├── models/               # Contains YOLOv5 model (best.pt)
├── videos/               # Input videos (tacticam.mp4, broadcast.mp4)
├── outputs/
│   ├── crops/            # Detected player crops
│   └── annotated/        # Final annotated output videos
├── features/             # Numpy files with extracted feature embeddings
├── src/
│   ├── detect_players.py       # Detects players and crops them
│   ├── extract_features.py     # Extracts ResNet features from crops
│   ├── match_players.py        # Performs cosine similarity matching
│   ├── visualize.py            # Annotates video with matched player IDs
│   └── evaluate.py             # Evaluates matching accuracy
├── main.py               # Runs full pipeline
├── streamlit_app.py      # Optional frontend using Streamlit
├── requirements.txt      # Required Python packages
└── README.md             # This file
🚀 How to Run
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/tanmay7175/MatchVision.git
cd MatchVision
Set Up Environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
Add Input Files:

Place tacticam.mp4 and broadcast.mp4 in the videos/ folder.

Add your trained YOLO model (best.pt) in the models/ folder.

Run Full Pipeline:

bash
Copy
Edit
python main.py
🧠 Model Architecture
Detection: YOLOv5 (best.pt)

Embedding Extraction: ResNet pretrained on ImageNet

Matching: Cosine similarity between player features

📈 Evaluation
Match accuracy is computed by comparing predicted player matches with a ground truth CSV. Accuracy is reported as a percentage.

💡 Future Enhancements
Add support for real-time processing using webcam/live stream.

Improve feature robustness using Siamese or Triplet Networks.

Include pose or temporal tracking information for better accuracy.

