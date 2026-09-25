import os
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

# Base directory
BASE_DIR = "/path/to/Requirements/dataset_gold"

# Input folder containing the original 5 CSV files
INPUT_DIR = os.path.join(BASE_DIR, "Experiment_dataset")

# Output folders
OUTPUT_DIRS = {
    2: os.path.join(BASE_DIR, "Binary_dataset"),
    3: os.path.join(BASE_DIR, "Tertiary_dataset"),
    4: os.path.join(BASE_DIR, "Quaternary_dataset"),
    5: os.path.join(BASE_DIR, "Quinary_dataset")
}

# Number of samples to take from each selected class
SAMPLES_PER_CLASS = {
    2: 150,   # Binary
    3: 100,   # Tertiary
    4: 75,    # Quaternary
    5: 60     # Quinary
}

# Fixed random seed for reproducibility
# Running the script again will create exactly the same dataset
RANDOM_SEED = 42


# ==========================================================
# Dataset Creation Function
# ==========================================================

def create_dataset(selected_classes):
    """
    selected_classes Example:
        ["health", "energy"]
        ["health", "energy", "safety"]
        ["health", "energy", "safety", "other"]
        ["health", "energy", "safety", "entertainment", "other"]
    """

    num_classes = len(selected_classes)

    if num_classes not in [2, 3, 4, 5]:
        raise ValueError("Select exactly 2, 3, 4 or 5 classes.")

    samples_needed = SAMPLES_PER_CLASS[num_classes]

    all_data = []

    print("=" * 60)
    print("Creating Dataset")
    print("=" * 60)

    for cls in selected_classes:

        FILE_NAMES = {
    "health": "Health.csv",
    "energy": "Energy.csv",
    "safety": "Safety.csv",
    "entertainment": "Entertainment.csv",
    "other": "Other.csv"
}

        file_path = os.path.join(INPUT_DIR, FILE_NAMES[cls.lower()])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"\nFile not found:\n{file_path}")

        df = pd.read_csv(file_path)

        if len(df) < samples_needed:
            raise ValueError(
                f"{cls}.csv contains only {len(df)} rows."
                f"\nNeed at least {samples_needed} rows."
            )

        # Deterministic random sampling
        sampled_df = df.sample(
            n=samples_needed,
            random_state=RANDOM_SEED
        )

        all_data.append(sampled_df)

        print(f"{cls.capitalize():15s} -> {samples_needed} samples selected")

    # Merge
    final_df = pd.concat(all_data, ignore_index=True)

    # Deterministic shuffle
    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # Dataset type
    dataset_type = {
        2: "binary",
        3: "tertiary",
        4: "quaternary",
        5: "quinary"
    }[num_classes]

    # Output folder
    output_folder = OUTPUT_DIRS[num_classes]
    os.makedirs(output_folder, exist_ok=True)

    # File name
    class_part = "_".join([x.capitalize() for x in selected_classes])

    filename = f"{class_part}_{dataset_type}_exp.csv"

    save_path = os.path.join(output_folder, filename)

    # Save CSV
    final_df.to_csv(save_path, index=False)

    print("\nDataset Successfully Created")
    print(f"Total Samples : {len(final_df)}")
    print(f"Saved At      : {save_path}")
    print("=" * 60)


# ==========================================================
# Examples
# Uncomment ONLY the dataset you want to generate.
# ==========================================================

# ---------------- Binary ----------------
#create_dataset([
#    "health",
 #   "energy"
#])

# ---------------- Tertiary ----------------
#create_dataset([
 #   "health",
  #  "energy",
   # "other"
#])

# ---------------- Quaternary ----------------
#create_dataset([
 #    "health",
  #   "entertainment",
   #"safety",
    #"energy"
# ])

# ---------------- Quinary ----------------
create_dataset([
   "health",
    "energy",
    "safety",
    "entertainment",
    "other"
])
