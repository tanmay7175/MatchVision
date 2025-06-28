import csv

def generate_ground_truth(matches, output_csv_path):
    """
    Auto-generate a dummy ground truth file for demonstration.
    Each matched pair is treated as ground truth.
    """
    with open(output_csv_path, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tacticam", "broadcast"])  # header

        for tacticam_file, broadcast_file in matches:
            writer.writerow([tacticam_file, broadcast_file])
