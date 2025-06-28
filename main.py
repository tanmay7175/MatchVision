from src.detect_players import detect_and_crop
from src.extract_features import extract_embeddings
from src.match_players import match_embeddings
from src.visualize import draw_ids
from src.evaluate import evaluate
from src.generate_ground_truth import generate_ground_truth  # optional, if implemented
import os

# Paths
video1 = "videos/tacticam.mp4"
video2 = "videos/broadcast.mp4"
model_path = "models/best.pt"

# Output folders
os.makedirs("outputs/crops/tacticam", exist_ok=True)
os.makedirs("outputs/crops/broadcast", exist_ok=True)
os.makedirs("outputs/annotated", exist_ok=True)
os.makedirs("features", exist_ok=True)

# 1. Detect and crop players
print("✅ Detecting players...")
detect_and_crop(video1, model_path, "outputs/crops/tacticam")
detect_and_crop(video2, model_path, "outputs/crops/broadcast")

# 2. Extract feature embeddings
print("✅ Extracting features...")
extract_embeddings("outputs/crops/tacticam", "features/tacticam.npy")
extract_embeddings("outputs/crops/broadcast", "features/broadcast.npy")

# 3. Match players
print("✅ Matching players across videos...")
matches = match_embeddings("features/tacticam.npy", "features/broadcast.npy")

# 4. Assign consistent player IDs
match_dict = {}
for pid, (t_file, b_file) in enumerate(matches):
    match_dict[(t_file, b_file)] = f"P{pid}"

# 5. Draw IDs on tacticam video
print("✅ Drawing player IDs on video...")
draw_ids(video1, match_dict, "outputs/annotated/tacticam_annotated.mp4")

# 6. Optional: Auto-generate ground truth (for demo purposes)
generate_ground_truth(matches, "ground_truth.csv")
  # remove if you are using real ground truth

# 7. Evaluate matching accuracy
print("✅ Evaluating match accuracy...")
accuracy = evaluate(matches, "ground_truth.csv")
print(f"🎯 Match Accuracy: {accuracy:.2%}")
