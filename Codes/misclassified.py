import pandas as pd
import os

# ============================================================
# CHANGE ONLY THIS: path to your highest F1 file
# ============================================================
INPUT_FILE = '/path/to/Requirements/generated_result5/Qwen2.5 72B_Health_Other_Safety_Entertainment_Energy_FSR_HF1.csv'

# Output directory (your specified path)
OUTPUT_DIR = '/path/to/Requirements/Error Analysis'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'misclassified_samples_quinary_small_Hf1.csv')

# ============================================================
# EXTRACT MISCLASSIFIED — NOTHING ELSE
# ============================================================
df = pd.read_csv(INPUT_FILE)

# Keep only wrong predictions
mis = df[df['true_label'] != df['predicted_label']].copy()

# Add label pair
mis['label_pair'] = mis['true_label'] + ' -> ' + mis['predicted_label']

# Sort for easier manual review
mis = mis.sort_values(['label_pair', 'req_id']).reset_index(drop=True)

# Save
mis.to_csv(OUTPUT_FILE, index=False)

print(f"Total samples      : {len(df)}")
print(f"Misclassified      : {len(mis)}")
print(f"Saved to           : {OUTPUT_FILE}")
print(f"\nTop error patterns:")
print(mis['label_pair'].value_counts().head())
