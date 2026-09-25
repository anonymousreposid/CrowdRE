# ============================================================
# CELL 1: NLTK Setup
# ============================================================

import nltk
import os
import subprocess

NLTK_DATA_DIR = r"/path/to/Requirements/nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

packages = [
    'punkt', 'punkt_tab', 'stopwords',
    'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
    'wordnet', 'omw-1.4'
]
for p in packages:
    nltk.download(p, download_dir=NLTK_DATA_DIR, quiet=True)

# ============================================================
# CELL 2: Configuration
# ============================================================
import sys
import re
import time
import pandas as pd

# ============================================================
# *** CHANGE ONLY THESE LINES FOR ANY NEW MODEL / TASK ***
# ============================================================

# MODEL_NAME     = "gemma3_27B"
# MODEL_NAME     = "Qwen2.5_72B"
# MODEL_NAME     = "Mistral 8x22B_4bit"
MODEL_NAME     = "Llama3.3_70B"

# MODEL_PATH_HF  = "unsloth/gemma-3-27b-it-GGUF"
# MODEL_PATH_HF  = "bartowski/Qwen2.5-72B-Instruct-GGUF"
# MODEL_PATH_HF  = "MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF"
MODEL_PATH_HF  = "MaziyarPanahi/Llama-3.3-70B-Instruct-GGUF"

#MODEL_FILENAME = "gemma-3-27b-it-Q4_K_M.gguf"
#MODEL_FILENAME = "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
# MODEL_FILENAME =[
#     "Mixtral-8x22B-v0.1.Q4_K_M-00001-of-00005.gguf",
#     "Mixtral-8x22B-v0.1.Q4_K_M-00002-of-00005.gguf",
#     "Mixtral-8x22B-v0.1.Q4_K_M-00003-of-00005.gguf",
#     "Mixtral-8x22B-v0.1.Q4_K_M-00004-of-00005.gguf",
#     "Mixtral-8x22B-v0.1.Q4_K_M-00005-of-00005.gguf",
# ]
MODEL_FILENAME = "Llama-3.3-70B-Instruct.Q4_K_M.gguf"


SELECTED_CLASSES = ["Entertainment", "Other", "Safety", "Health"]
SAMPLES_PER_CLASS = None

UNCLASSIFIED = "UNCLASSIFIED"

CLASS_SUFFIX = "_".join(SELECTED_CLASSES)
TASK_TYPE = {2: "binary", 3: "tertiary", 4: "quaternary", 5: "quinary"}.get(
    len(SELECTED_CLASSES), f"{len(SELECTED_CLASSES)}-way"
)

SHARD_FILES = MODEL_FILENAME if isinstance(MODEL_FILENAME, list) else [MODEL_FILENAME]
# ============================================================

BASE_DIR       = r"/path/to/Requirements"
MODEL_DIR      = os.path.join(BASE_DIR, "models")
LOCAL_GOLD_CSV = os.path.join(BASE_DIR, "dataset_gold","Quaternary_dataset","Health_Entertainment_Safety_Other_quaternary_exp.csv")
OUT_DIR        = os.path.join(BASE_DIR, "generated_result4")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ALL_CLASS_DEFINITIONS = {
    "Energy":        "Monitoring, controlling, optimizing, managing energy consumption or energy-related resources.",
    "Entertainment": "Media consumption, content delivery, audio/video systems, gaming, or user entertainment experiences.",
    "Health":        "Physical health, healthcare monitoring, medication management, fitness, or well-being.",
    "Safety":        "Protection of users, property, or assets from hazards, accidents, intrusions, emergencies, or security threats.",
    "Other":         "General home automation, convenience, communication, information management, device coordination, or other smart home functions.",
}

ALL_FSR_EXAMPLES = {
    "Energy": [
        {
            "req_id": "R1",
            "requirement": "shut the stove off for me, I can work on other things while the food is cooking.",
            "label": "Energy",
            "explanation": "This requirement is classified as Energy because it involves controlling appliance operation to avoid unnecessary energy use while cooking. The functionality supports more efficient management of energy-consuming devices in the home."
        },
        {
            "req_id": "R2",
            "requirement": "my smart home to adjust the temperature based on the weather outside, I can save on my energy bill by not using the AC unnecessarily.",
            "label": "Energy",
            "explanation": "This requirement is classified as Energy because it optimizes temperature settings according to external weather conditions to reduce unnecessary air conditioning use. The primary benefit is lowering energy consumption and decreasing energy costs."
        },
    ],
    "Safety": [
        {
            "req_id": "R3",
            "requirement": "my smart home to turn on certain lights at dusk, I can feel comfortable knowing that they made it home safely that day.",
            "label": "Safety",
            "explanation": "This requirement is classified as Safety because it aims to provide a well-lit environment during low-light conditions to enhance security and reduce concerns about occupants arriving home safely. The primary benefit is protecting household members and increasing their sense of security."
        },
        {
            "req_id": "R4",
            "requirement": "my smart home to keep me up to date about my children's activities when I'm out of the home, I can know they're safe and positively occupied",
            "label": "Safety",
            "explanation": "This requirement is classified as Safety because it focuses on monitoring children's activities and providing updates that help ensure their well-being while the user is away. The primary objective is to support awareness and protection of household members."
        },
    ],
    "Entertainment": [
        {
            "req_id": "R5",
            "requirement": "my smart home to play my favourite music playlist automatically when I arrive home, so I can relax immediately after work.",
            "label": "Entertainment",
            "explanation": "This requirement is classified as Entertainment because it focuses on automated media playback to enhance the user's leisure experience upon arriving home. The primary objective is content delivery and personal enjoyment."
        },
        {
            "req_id": "R6",
            "requirement": "my smart home to dim the lights and start the movie on my TV when I say movie time, so I can have a cinema-like experience at home.",
            "label": "Entertainment",
            "explanation": "This requirement is classified as Entertainment because it creates an immersive media consumption environment by coordinating lighting and video playback. The primary benefit is enhancing the user's home entertainment experience."
        },
    ],
    "Health": [
        {
            "req_id": "R7",
            "requirement": "my smart home to remind me to take my medication at the scheduled times every day, so I don't miss any doses.",
            "label": "Health",
            "explanation": "This requirement is classified as Health because it addresses medication management and adherence to prescribed schedules. The primary objective is supporting the user's physical health through timely reminders."
        },
        {
            "req_id": "R8",
            "requirement": "my smart home to track my daily step count and suggest when I should go for a walk, so I can maintain an active lifestyle.",
            "label": "Health",
            "explanation": "This requirement is classified as Health because it involves fitness tracking and activity recommendations to promote physical well-being. The primary benefit is encouraging regular physical activity for better health outcomes."
        },
    ],
    "Other": [
        {
            "req_id": "R9",
            "requirement": "my smart home to send me a notification when a package is delivered to my front door, so I can retrieve it promptly.",
            "label": "Other",
            "explanation": "This requirement is classified as Other because it involves general communication and notification functionality for package delivery. The primary objective is convenience and information management rather than energy, health, safety, or entertainment."
        },
        {
            "req_id": "R10",
            "requirement": "my smart home to automatically create a grocery list based on items running low in the kitchen, so I can shop efficiently.",
            "label": "Other",
            "explanation": "This requirement is classified as Other because it supports general household management through automated list generation. The primary objective is convenience and device coordination for everyday tasks."
        },
    ],
}

print(f"Task: {' vs '.join(SELECTED_CLASSES)}")
print(f"Task type: {TASK_TYPE}")
print(f"Number of classes: {len(SELECTED_CLASSES)}")
print(f"Samples per class: {SAMPLES_PER_CLASS if SAMPLES_PER_CLASS else 'ALL'}")
print(f"Model repo: {MODEL_PATH_HF}")
print(f"Model file(s): {SHARD_FILES}")
print(f"Dataset: {LOCAL_GOLD_CSV}")
print(f"Output: {OUT_DIR}")

# ============================================================
# CELL 3: Download Model
# ============================================================
from huggingface_hub import hf_hub_download

all_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in SHARD_FILES)

if all_exist:
    print(f"All {len(SHARD_FILES)} model file(s) already downloaded.")
    for f in SHARD_FILES:
        fpath = os.path.join(MODEL_DIR, f)
        print(f"  {f} ({os.path.getsize(fpath) / 1e9:.1f} GB)")
else:
    print(f"Downloading {len(SHARD_FILES)} file(s) from {MODEL_PATH_HF}...")
    for shard in SHARD_FILES:
        fpath = os.path.join(MODEL_DIR, shard)
        if os.path.exists(fpath):
            print(f"  {shard} already exists ({os.path.getsize(fpath) / 1e9:.1f} GB)")
            continue
        print(f"  Downloading {shard}...")
        hf_hub_download(
            repo_id=MODEL_PATH_HF,
            filename=shard,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  {shard} done")

MODEL_PATH = os.path.join(MODEL_DIR, SHARD_FILES[0])
print(f"\nModel path (first file): {MODEL_PATH}")

total_size = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e9
                 for f in SHARD_FILES if os.path.exists(os.path.join(MODEL_DIR, f)))
print(f"Total model size: {total_size:.1f} GB")

# ============================================================
# CELL 4: Verify Dataset
# ============================================================
if not os.path.exists(LOCAL_GOLD_CSV):
    raise FileNotFoundError(f"Dataset not found: {LOCAL_GOLD_CSV}")
print(f"Dataset file found: {LOCAL_GOLD_CSV}")

# ============================================================
# CELL 5: Load Dataset
# ============================================================
gold_full = pd.read_csv(LOCAL_GOLD_CSV)
print(f"Dataset loaded: {len(gold_full)} rows")
print(f"Columns: {list(gold_full.columns)}")

# Drop rows with empty/NaN labels
gold_full = gold_full.dropna(subset=['label']).reset_index(drop=True)
print(f"After dropping NaN labels: {len(gold_full)} rows")

available_labels = set(gold_full['label'].unique())
missing = set(SELECTED_CLASSES) - available_labels
if missing:
    raise ValueError(f"Classes not found: {missing}. Available: {available_labels}")

if SAMPLES_PER_CLASS:
    task_data = gold_full[gold_full['label'].isin(SELECTED_CLASSES)].groupby('label').apply(
        lambda x: x.sample(n=min(SAMPLES_PER_CLASS, len(x)), random_state=42)
    ).reset_index(drop=True)
else:
    task_data = gold_full[gold_full['label'].isin(SELECTED_CLASSES)].copy()

task_data = task_data.sample(frac=1, random_state=42).reset_index(drop=True)
gold = task_data.copy()

# Ensure reference_explanation column exists
if 'reference_explanation' not in gold.columns:
    gold['reference_explanation'] = ""
    task_data['reference_explanation'] = ""
    print("WARNING: 'reference_explanation' column not found in dataset, using empty strings")

print(f"\nTask type: {TASK_TYPE} ({len(SELECTED_CLASSES)} classes)")
print(f"Total samples: {len(task_data)}")
for cls in SELECTED_CLASSES:
    print(f"  {cls}: {len(task_data[task_data['label'] == cls])}")

# ============================================================
# CELL 6: Load LLM
# ============================================================

LLAMA_CLI = "/path/to/Requirements/llama.cpp/build/bin/llama-cli"

print("Using llama-cli backend.")

# ============================================================
# CELL 7: Prompts
# ============================================================
CLASS_DEFS_TEXT = "\n".join([
    f"{cls}: {ALL_CLASS_DEFINITIONS[cls]}" for cls in ALL_CLASS_DEFINITIONS
])
AVAILABLE_CLASSES_TEXT = "{" + ", ".join(SELECTED_CLASSES) + "}"


def build_fsr_examples_text():
    example_lines = []
    for cls in SELECTED_CLASSES:
        if cls in ALL_FSR_EXAMPLES:
            for ex in ALL_FSR_EXAMPLES[cls]:
                example_lines.append(
                    f"Req ID: {ex['req_id']}\n"
                    f"Requirement: {ex['requirement']}\n"
                    f"Label: {ex['label']}\n"
                    f"Explanation: {ex['explanation']}"
                )
    return "\n\n".join(example_lines)


FSR_EXAMPLES_TEXT = build_fsr_examples_text()


def ezs_prompt(req_id, req):
    return f"""You are an expert requirements analyst for smart home applications. Your task is to classify a software requirement into exactly one available class label and provide a concise explanation for the assigned label.

{CLASS_DEFS_TEXT}

Available Labels for this Classification Task:
{AVAILABLE_CLASSES_TEXT}

Rules:
Assign exactly one class label from the available labels.
Use the class label definitions to determine the most appropriate label.
Base the classification and explanation only on information explicitly stated or strongly implied in the requirement.
Do not hallucinate.
If multiple labels appear relevant, select the label that best represents the requirement's primary objective.
Do not mention, compare, or reference any class label other than the assigned label in the explanation.
Focus on the requirement's primary objective, functionality, and user benefit.
Generate a concise explanation in exactly only in two sentences.

You MUST respond in this exact format:
Label: <LABEL>
Explanation: This requirement is classified as [LABEL] because...

Now classify this requirement:
Req ID: {req_id}
Req: {req}
"""


def fsr_prompt(req_id, req):
    return f"""You are an expert requirements analyst for smart home applications. Your task is to classify a software requirement into exactly one class label and provide a concise explanation for the assigned label.

{CLASS_DEFS_TEXT}

Available Labels for This Classification Task:
{AVAILABLE_CLASSES_TEXT}

Rules:
Assign exactly one class label from the available labels.
Use the class label definitions to determine the most appropriate label.
Base the classification and explanation only on information explicitly stated or strongly implied in the requirement.
Do not hallucinate.
If multiple labels appear relevant, select the label that best represents the requirement's primary objective.
Do not mention, compare, or reference any class label other than the assigned label in the explanation.
Focus on the requirement's primary objective, functionality, and user benefit.
Generate a concise explanation in exactly two sentences.

Examples:
{FSR_EXAMPLES_TEXT}

You MUST respond in this exact format:
Label: <LABEL>
Explanation: This requirement is classified as [LABEL] because...

Now classify this requirement:
Req ID: {req_id}
Requirement: {req}
"""


def fcl_prompt(req_id, req):
    return f"""You are an expert requirements analyst for smart home applications. Your task is to classify a software requirement into exactly one class label and provide a concise explanation for the assigned label.

{CLASS_DEFS_TEXT}

Available Labels for This Classification Task:
{AVAILABLE_CLASSES_TEXT}

Task:
Step 1: Identify three key facts from the requirement that support classification.
Step 2: Use the identified facts and the class label definitions to determine the most appropriate class label from the available labels.
Step 3: Generate a concise explanation for the assigned label.

Rules:
Assign exactly one class label from the available labels.
Use the class label definitions to determine the most appropriate label.
Base the facts, classification, and explanation only on information explicitly stated or strongly implied in the requirement.
Do not hallucinate.
If multiple labels appear relevant, select the label that best represents the requirement's primary objective.
Do not mention, compare, or reference any class label other than the assigned label in the explanation.
Focus on the requirement's primary objective, functionality, and user benefit.
Each fact must be a single sentence.
Generate a concise explanation in exactly two sentences.

You MUST respond in this exact format:
Label: <LABEL>
Explanation: This requirement is classified as [LABEL] because...

Now classify this requirement:
Req ID: {req_id}
Req: {req}
"""


PROMPTS = {"EZS": ezs_prompt, "FSR": fsr_prompt, "FCL": fcl_prompt}

# ============================================================
# CELL 8: Parser & Generator
# ============================================================

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\-_]|$$[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def clean_llama_output(text, prompt):
    if not text:
        return ""
    text = strip_ansi(text)
    lines = text.split('\n')
    cleaned_lines = []
    found_content = False
    for line in lines:
        stripped = line.strip()
        if not found_content:
            if stripped == '' or set(stripped).issubset({'|', '-', '\\', '/', ' ', '>', '<'}):
                continue
            if any(stripped.startswith(x) for x in [
                'build:', 'model:', 'available', '/', 'Script',
                'Loading model', 'ggml_cuda_init', 'llama_model_loader',
                'llm_load', 'system_info:', 'main:', 'generate:', 'sampling:'
            ]):
                continue
            found_content = True
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    if prompt in text:
        text = text.split(prompt, 1)[-1]
    else:
        prompt_last_line = prompt.split('\n')[-1].strip()
        if prompt_last_line and prompt_last_line in text:
            last_idx = text.rfind(prompt_last_line)
            text = text[last_idx + len(prompt_last_line):]

    text = re.sub(r"\[\s*Prompt:.*?$$", "", text, flags=re.DOTALL)
    text = re.sub(r"$$\s*Start thinking$$.*?$$\s*End thinking$$", "", text, flags=re.DOTALL)
    text = re.sub(r"Exiting\.+\s*", "", text)
    text = re.sub(r">\s*$", "", text)
    text = re.sub(r"Script done on.*", "", text, flags=re.DOTALL)

    last_label_pos = text.rfind("Label:")
    if last_label_pos != -1:
        text = text[last_label_pos:]

    return text.strip()


def log_fallback_instant(req_id, prompt_name, fallback_type,
                         predicted_label_found, raw_output, out_dir):
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n  >>> FALLBACK [{timestamp}] | {prompt_name} | {req_id} | {fallback_type} <<<")

    record = {
        "timestamp": timestamp,
        "req_id": req_id,
        "prompt_name": prompt_name,
        "fallback_type": fallback_type,
        "predicted_label_found": predicted_label_found if predicted_label_found else "",
        "raw_output": (raw_output[:500] + "...") if len(raw_output) > 500 else raw_output
    }

    fallback_file = os.path.join(
        out_dir, f"RUNNING_FALLBACK_{MODEL_NAME}_{TASK_TYPE}_{CLASS_SUFFIX}_{prompt_name}.csv"
    )
    file_exists = os.path.exists(fallback_file)
    pd.DataFrame([record]).to_csv(
        fallback_file, mode='a', header=not file_exists,
        index=False, encoding='utf-8'
    )
    return record


def _extract_explanation(text):
    """Extract whatever explanation the model produced.
    Works on text already truncated to the last Label: block.
    """
    expl = ""

    expl_patterns = [
        r'Explanation:\s*(.*?)(?=\n\s*(?:Label:|Req ID:|Step \d|Predicted|Class:|\Z))',
        r'Generated Explanation:\s*(.*?)(?=\n\s*(?:Label:|Req ID:|Step \d|Predicted|Class:|\Z))',
        r'(This requirement is classified as.*?)(?=\n\s*(?:Label:|Req ID:|Step \d|Predicted|Class:|\Z))',
    ]
    for pat in expl_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            expl = m.group(1).strip()
            break

    expl = re.sub(r'^Explanation:\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^Generated Explanation:\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^(?:Label|Class|Predicted Class):\s*[^:]+:\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^This requirement is classified as\s*[^:]+:\s*because\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^It is classified as\s*[^:]+:\s*because\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^The requirement is classified as\s*[^:]+:\s*because\s*', '', expl, flags=re.IGNORECASE).strip()
    expl = re.sub(r'^because\s+', '', expl, flags=re.IGNORECASE).strip()
    expl = re.split(r'\n\s*(?:$$|>|Exiting|Script|Label:|Req ID:)', expl)[0].strip()

    if not expl or len(expl) < 5:
        last_label_pos = text.rfind("Label:")
        if last_label_pos != -1:
            raw_section = text[last_label_pos:].strip()
            raw_section = re.sub(r'^Label:\s*\S+\s*', '', raw_section, flags=re.IGNORECASE).strip()
            raw_section = re.sub(r'^Explanation:\s*', '', raw_section, flags=re.IGNORECASE).strip()
            raw_section = re.sub(r'^This requirement is classified as\s*[^:]+:\s*because\s*', '', raw_section, flags=re.IGNORECASE).strip()
            raw_section = re.sub(r'^because\s+', '', raw_section, flags=re.IGNORECASE).strip()
            if len(raw_section) >= 5:
                expl = raw_section

    expl = expl[:500].strip()
    expl = (expl.replace('\u2018', "'").replace('\u2019', "'")
                 .replace('\u201c', '"').replace('\u201d', '"')
                 .replace('\u2013', '-').replace('\u2014', '-'))
    return expl


def _normalize_explanation_format(expl, label):
    """Convert any explanation style to:
    'This requirement is classified as [LABEL] because ...'"""
    if not expl or len(expl) < 5:
        return expl

    # Already correct format — do nothing
    pattern = r'^This requirement is classified as\s+\w+\s+because'
    if re.match(pattern, expl, re.IGNORECASE):
        return expl

    # Model wrote something else — wrap it
    cleaned = re.sub(r'^This requirement\s+', '', expl, flags=re.IGNORECASE).strip()
    if cleaned:
        cleaned = cleaned[0].lower() + cleaned[1:]

    return f"This requirement is classified as {label} because {cleaned}"


def parse_output(text, req_id=None, prompt_name=None, out_dir=None):
    """Extract label and explanation from model output.

    Flow:
      1. Valid label in SELECTED_CLASSES?     -> (label, explanation)
      2. Invalid label (e.g. "SmartHome")?    -> ("SmartHome", explanation)
      3. No label pattern at all?             -> (UNCLASSIFIED, explanation)
      4. Empty/garbage output?                -> (UNCLASSIFIED, "")

    Explanation is always extracted from the same text block.
    """
    try:
        if not text or len(text.strip()) < 5:
            log_fallback_instant(req_id, prompt_name, "Empty/Short Output",
                                 None, repr(text), out_dir)
            return UNCLASSIFIED, ""

        valid_patterns = [
            rf'Label:\s*({"|".join(SELECTED_CLASSES)})',
            rf'Predicted Class:\s*({"|".join(SELECTED_CLASSES)})',
            rf'Class:\s*({"|".join(SELECTED_CLASSES)})',
            rf'classified as\s+({"|".join(SELECTED_CLASSES)})',
            rf'({"|".join(SELECTED_CLASSES)})\s+is the most appropriate',
            rf'({"|".join(SELECTED_CLASSES)})\s+is the best',
        ]

        label = None
        for pat in valid_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                label = m.group(1).strip()
                break

        if label in SELECTED_CLASSES:
            expl = _extract_explanation(text)
            expl = _normalize_explanation_format(expl, label)

            if not expl or len(expl) < 5:
                log_fallback_instant(req_id, prompt_name, "Explanation Missing",
                                     label, text, out_dir)
            return label, expl

        any_label_patterns = [
            r'Label:\s*(\S+)',
            r'Predicted Class:\s*(\S+)',
            r'Class:\s*(\S+)',
        ]

        raw_label = None
        for pat in any_label_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw_label = m.group(1).strip()
                raw_label = re.sub(r'[:\.,;]+', '', raw_label).strip()
                break

        expl = _extract_explanation(text)

        if raw_label:
            log_fallback_instant(req_id, prompt_name, "Invalid Label",
                                 raw_label, text, out_dir)
            return raw_label, expl

        log_fallback_instant(req_id, prompt_name, "No Label Found",
                             None, text, out_dir)
        return UNCLASSIFIED, expl

    except Exception as e:
        log_fallback_instant(req_id, prompt_name,
                             f"Parse Exception: {str(e)}", None,
                             text[:1000] if text else "N/A", out_dir)
        return UNCLASSIFIED, ""


def generate(prompt):
    try:
        cmd = [
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-ngl", "999",
            "-n", "256",
            "--ctx-size", "4096",
            "--single-turn",
            "-no-cnv",
            "--no-display-prompt",
            "--simple-io",
            "-lv", "0",
            "--temp", "0.0",
            "--top-p", "1.0",
            "-p", prompt,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True,
        )

        text = result.stdout
        if not text.strip() and result.stderr:
            text = result.stderr

        text = clean_llama_output(text, prompt)
        return text.strip()

    except subprocess.CalledProcessError as e:
        err = e.stderr[:500] if e.stderr else 'unknown'
        print(f"  llama-cli error: {err}")
        return ""
    except Exception as e:
        print(f"  generate() exception: {e}")
        return ""


# ============================================================
# CELL 9: Main Loop (WITH RESUME SUPPORT)
# ============================================================

DEBUG_RAW = False

for prompt_name, prompt_fn in PROMPTS.items():
    pred_file = os.path.join(OUT_DIR, f"{MODEL_NAME}_{CLASS_SUFFIX}_{prompt_name}_HF1.csv")
    checkpoint_file = os.path.join(OUT_DIR, f"{MODEL_NAME}_{CLASS_SUFFIX}_{prompt_name}_HF1_checkpoint.csv")
    debug_file = os.path.join(OUT_DIR, f"{MODEL_NAME}_{CLASS_SUFFIX}_{prompt_name}_debug.txt")

    # --- RESUME LOGIC: Check if already completed ---
    if os.path.exists(pred_file):
        existing_df = pd.read_csv(pred_file)
        if len(existing_df) == len(task_data):
            print(f"Skipping {prompt_name} -- fully completed: {pred_file}")
            continue
        elif len(existing_df) > 0:
            # Partial completion - find where to resume
            last_processed_req_id = existing_df['req_id'].iloc[-1]
            print(f"\nResuming {prompt_name} from after req_id: {last_processed_req_id}")
            print(f"Already processed: {len(existing_df)}/{len(task_data)} rows")
            
            # Create a set of already processed req_ids for fast lookup
            processed_req_ids = set(existing_df['req_id'].tolist())
            
            # Load existing predictions to append to
            preds = existing_df.to_dict('records')
        else:
            preds = []
            processed_req_ids = set()
    else:
        preds = []
        processed_req_ids = set()

    # Also check checkpoint file if pred_file doesn't exist fully
    if not preds and os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        if len(checkpoint_df) > 0:
            last_processed_req_id = checkpoint_df['req_id'].iloc[-1]
            print(f"\nResuming {prompt_name} from checkpoint after req_id: {last_processed_req_id}")
            print(f"Checkpoint has: {len(checkpoint_df)} rows")
            processed_req_ids = set(checkpoint_df['req_id'].tolist())
            preds = checkpoint_df.to_dict('records')

    print(f"\n{'='*60}")
    print(f">>> {MODEL_NAME} | {TASK_TYPE} ({len(SELECTED_CLASSES)} classes) | {' vs '.join(SELECTED_CLASSES)} | {prompt_name} <<<")
    print(f"{'='*60}")

    debug_logs = []
    start = time.time()

    for idx, row in task_data.iterrows():
        req_id, req, true_l = row['req_id'], row['requirement'], row['label']
        
        # --- SKIP ALREADY PROCESSED ROWS ---
        if req_id in processed_req_ids:
            continue  # Skip this row, already done
        
        prompt = prompt_fn(req_id, req)

        req_start = time.time()

        pred_l = UNCLASSIFIED
        best_expl = ""
        raw = ""

        for attempt in range(3):
            try:
                raw = generate(prompt)

                if idx < 3 and attempt == 0 and DEBUG_RAW:
                    debug_logs.append(
                        f"\n{'='*60}\nREQ: {req_id} | ATTEMPT: {attempt+1}\n"
                        f"{'='*60}\nRAW:\n{raw}\n{'='*60}"
                    )

                pred_l, expl = parse_output(raw, req_id, prompt_name, OUT_DIR)

                if expl and len(expl) > len(best_expl):
                    best_expl = expl

                if pred_l in SELECTED_CLASSES:
                    best_expl = expl
                    break
                else:
                    if attempt < 2:
                        print(f"  WARNING {req_id} attempt {attempt+1}/3: "
                              f"label='{pred_l}', retrying...")

            except Exception as e:
                if attempt < 2:
                    print(f"  WARNING {req_id} attempt {attempt+1}/3 exception: {e}")
                continue

        if pred_l not in SELECTED_CLASSES:
            log_fallback_instant(
                req_id, prompt_name,
                f"All Attempts Failed - Using Model Label: '{pred_l}'",
                pred_l, raw[:1000] if raw else "N/A", OUT_DIR
            )

        req_end = time.time()
        time_taken = round(req_end - req_start, 2)

        ref_expl = gold[gold['req_id'] == req_id]['reference_explanation'].values
        ref_expl = ref_expl[0] if len(ref_expl) > 0 else ""

        preds.append({
            'req_id': req_id,
            'requirement': req,
            'true_label': true_l,
            'predicted_label': pred_l,
            'generated_explanation': best_expl,
            'reference_explanation': ref_expl,
            'time_taken': time_taken,
        })

        # --- SAVE CHECKPOINT EVERY 10 ROWS ---
        if (len(preds) % 10) == 0:
            checkpoint_df = pd.DataFrame(preds)
            checkpoint_df = checkpoint_df[['req_id', 'requirement', 'true_label', 'predicted_label',
                                           'generated_explanation', 'reference_explanation', 'time_taken']]
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"  Checkpoint saved: {len(preds)} rows processed")

        if (len(preds) % 50) == 0 or idx == len(task_data) - 1:
            elapsed = time.time() - start
            avg = elapsed / (len(preds) if len(preds) > 0 else 1)
            remaining = len(task_data) - len(processed_req_ids) - len(preds) + len(processed_req_ids)
            eta = avg * remaining
            print(f"  {len(preds)}/{len(task_data)} | {avg:.1f}s/req | ETA: {eta/60:.1f}min")

    # --- FINAL SAVE ---
    pred_df = pd.DataFrame(preds)
    pred_df = pred_df[['req_id', 'requirement', 'true_label', 'predicted_label',
                       'generated_explanation', 'reference_explanation', 'time_taken']]
    pred_df.to_csv(pred_file, index=False, encoding='utf-8')

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Checkpoint cleaned up: {checkpoint_file}")
    
    total_time = round((time.time() - start) / 60, 2)
    print(f"  Saved: {pred_file} | Total: {total_time} min")

# ============================================================
# CELL 10: Done
# ============================================================
print("\n" + "=" * 60)
print("ALL GENERATION COMPLETE")
print(f"Task type: {TASK_TYPE} ({len(SELECTED_CLASSES)} classes)")
print(f"Results in: {OUT_DIR}")

# --- Prediction files ---
print("\n--- Prediction Files ---")
found_pred = False
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".csv") and not f.startswith("RUNNING_FALLBACK"):
        fpath = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        found_pred = True
        try:
            tmp_df = pd.read_csv(fpath)
            total = len(tmp_df)
            valid = len(tmp_df[tmp_df['predicted_label'].isin(SELECTED_CLASSES)])
            unclass = len(tmp_df[tmp_df['predicted_label'] == UNCLASSIFIED])
            invalid = total - valid - unclass
            status_parts = []
            if unclass > 0:
                status_parts.append(f"{unclass} UNCLASSIFIED")
            if invalid > 0:
                status_parts.append(f"{invalid} invalid-label")
            status = f" | {', '.join(status_parts)}" if status_parts else " | all valid"
            print(f"  {f}  ({size_kb:.1f} KB) [{valid}/{total} valid]{status}")
        except Exception:
            print(f"  {f}  ({size_kb:.1f} KB)")
if not found_pred:
    print("  (none)")

# --- Fallback log files ---
print("\n--- Fallback Log Files ---")
found_fb = False
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("RUNNING_FALLBACK") and f.endswith(".csv"):
        fpath = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        found_fb = True
        try:
            fb_df = pd.read_csv(fpath)
            fb_total = len(fb_df)
            # Count by fallback type
            type_counts = fb_df['fallback_type'].value_counts().to_dict()
            type_summary = ", ".join(f"{v}x {k}" for k, v in type_counts.items())
            print(f"  {f}  ({size_kb:.1f} KB) [{fb_total} fallbacks] -> {type_summary}")
        except Exception:
            print(f"  {f}  ({size_kb:.1f} KB)")
if not found_fb:
    print("  (none - no fallbacks occurred!)")

print("=" * 60)
