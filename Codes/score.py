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
from ctc_score.scorer import Scorer

import os
import re
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CELL 1: Paths
# ============================================================
BASE_DIR = r"/path/to/Requirements"
OUT_DIR  = os.path.join(BASE_DIR, "generated_result4")

# ============================================================
# CELL 2: CUDA Setup
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("WARNING: No CUDA GPU found. Running on CPU (will be slower).")

# ============================================================
# CELL 3: Classification Metrics
# ============================================================

UNCLASSIFIED = "UNCLASSIFIED"


def clf_metrics(y_true, y_pred, selected_classes):
    """Classification metrics — EVERY row counts.

    Logic:
      predicted == true_label  -> CORRECT
      predicted != true_label  -> WRONG

    No row is ever filtered out. Invalid labels and UNCLASSIFIED
    count as wrong predictions.

    For per-class precision/recall/F1, predictions outside selected_classes
    are treated as "other" — they contribute to false negatives for every
    selected class and false positives for none.
    """
    y_true = [str(t) for t in y_true]
    y_pred = [str(p) for p in y_pred]

    total = len(y_true)
    if total == 0:
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'macro_f1': 0.0,
            'total': 0, 'correct': 0, 'wrong': 0,
            'valid_correct': 0, 'valid_wrong': 0,
            'invalid_predicted': 0, 'unclassified': 0,
        }

    # ── Overall accuracy: every row counts ────────────────────
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    wrong = total - correct
    accuracy = correct / total

    # ── Prediction breakdown ──────────────────────────────────
    valid_correct = 0
    valid_wrong = 0
    invalid_predicted = 0
    unclassified_count = 0

    for t, p in zip(y_true, y_pred):
        if p == UNCLASSIFIED:
            unclassified_count += 1
        elif p not in selected_classes:
            invalid_predicted += 1
        elif t == p:
            valid_correct += 1
        else:
            valid_wrong += 1

    #    Use sklearn with all labels (selected + any predicted extras)
    #    so confusion matrix is complete
    all_labels = sorted(set(y_true) | set(y_pred))

    # For macro metrics, compute only over selected_classes
    per_class = {}
    for cls in selected_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'support': sum(1 for t in y_true if t == cls),
        }

    macro_p  = np.mean([v['precision'] for v in per_class.values()])
    macro_r  = np.mean([v['recall'] for v in per_class.values()])
    macro_f1 = np.mean([v['f1'] for v in per_class.values()])

    return {
        'accuracy':         round(accuracy, 4),
        'precision':        round(macro_p, 4),
        'recall':           round(macro_r, 4),
        'macro_f1':         round(macro_f1, 4),
        'total':            total,
        'correct':          correct,
        'wrong':            wrong,
        'valid_correct':    valid_correct,
        'valid_wrong':      valid_wrong,
        'invalid_predicted': invalid_predicted,
        'unclassified':     unclassified_count,
    }


# ============================================================
# CELL 4: CTC Preservation Metric (OFFICIAL ctc_score library)
# ============================================================


class StyleTransferScorer(Scorer):
    """Official StyleTransferScorer from ctc-gen-eval repo."""
    def __init__(self, align, aggr_type='mean', device='cuda'):
        Scorer.__init__(self, align=align, aggr_type=aggr_type, device=device)

    def score(self, input_sent, hypo, aspect,
              remove_stopwords=False, rescale_with_baseline=True):
        kwargs = dict(
            input_sent=input_sent,
            hypo=hypo,
            remove_stopwords=remove_stopwords,
            rescale_with_baseline=rescale_with_baseline)

        if aspect == 'preservation':
            return self.score_preservation(**kwargs)
        else:
            raise NotImplementedError

    def score_preservation(self, input_sent, hypo,
                           remove_stopwords, rescale_with_baseline):
        """Official preservation: harmonic mean of bidirectional alignments."""
        aligner = self._get_aligner('sent_to_sent')

        align_y_x = aligner.get_score(
            input_text=input_sent,
            context=hypo,
            remove_stopwords=remove_stopwords)

        align_x_y = aligner.get_score(
            input_text=hypo,
            context=input_sent,
            remove_stopwords=remove_stopwords)

        score = 2 * (align_y_x * align_x_y) / (align_y_x + align_x_y)

        if self._align.startswith('E') and rescale_with_baseline:
            baseline = aligner.baseline_vals[2].item()
            return (score - baseline) / (1 - baseline)

        return score


CTC_SCORER = None

def get_ctc_scorer(align='E-roberta', device='cuda'):
    global CTC_SCORER
    if CTC_SCORER is None:
        print(f"  Loading official CTC StyleTransferScorer ({align})...")
        CTC_SCORER = StyleTransferScorer(align=align, device=device)
    return CTC_SCORER


def preservation(y, x, device='cuda', rescale=True):
    scorer = get_ctc_scorer(device=device)
    return scorer.score(
        input_sent=x,
        hypo=y,
        aspect='preservation',
        remove_stopwords=False,
        rescale_with_baseline=rescale)


# ============================================================
# CELL 5: BERTScore
# ============================================================
from bert_score import score as bert_score_fn


def compute_bertscore(references, generated, device='cuda'):
    P, R, F1 = bert_score_fn(
        generated, references,
        lang='en',
        model_type='bert-base-uncased',
        device=device,
        verbose=True,
        batch_size=64,
    )
    return F1.mean().item()


# ============================================================
# CELL 5a: ROUGE-1, ROUGE-2, ROUGE-L and METEOR
# ============================================================
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


def compute_rouge_scores(references, generated):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rL_scores = [], [], []
    for ref, gen in zip(references, generated):
        try:
            scores = scorer.score(str(ref), str(gen))
            r1_scores.append(scores['rouge1'].fmeasure)
            r2_scores.append(scores['rouge2'].fmeasure)
            rL_scores.append(scores['rougeL'].fmeasure)
        except Exception:
            r1_scores.append(0.0)
            r2_scores.append(0.0)
            rL_scores.append(0.0)
    return np.mean(r1_scores), np.mean(r2_scores), np.mean(rL_scores)


def compute_meteor(references, generated):
    """Compute METEOR scores for generated vs reference explanations."""
    scores = []
    for ref, gen in zip(references, generated):
        try:
            s = meteor_score([str(ref).split()], str(gen).split())
            scores.append(s)
        except Exception:
            scores.append(0.0)
    return np.mean(scores)


# ============================================================
# CELL 6: Combined Explanation Metrics
# ============================================================

def exp_metrics(references, generated, device='cuda'):
    if not references or not generated:
        return {
            'preservation': 0.0, 'bertscore_f1': 0.0,
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0
        }

    print("  Computing CTC Preservation scores...")
    pres_scores = []
    for i, (y, x) in enumerate(zip(generated, references)):
        try:
            s = preservation(y, x, device=device, rescale=True)
            if hasattr(s, 'item'):
                s = float(s.item())
            else:
                s = float(s)
            pres_scores.append(s)
        except Exception as e:
            print(f"    Preservation error at {i}: {e}")
            pres_scores.append(0.0)
        if (i + 1) % 50 == 0:
            print(f"    Preservation: {i+1}/{len(generated)}")
    pres_mean = np.mean(pres_scores)

    print("  Computing BERTScore...")
    bert_f1 = compute_bertscore(references, generated, device=device)

    print("  Computing ROUGE-1, ROUGE-2, ROUGE-L...")
    rouge1, rouge2, rougeL = compute_rouge_scores(references, generated)

    print("  Computing METEOR...")
    meteor = compute_meteor(references, generated)

    return {
        'preservation': pres_mean,
        'bertscore_f1': bert_f1,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
        'meteor': meteor,
    }


# ============================================================
# CELL 7: Auto-detect & Evaluate ALL CSVs
# ============================================================

if not os.path.exists(OUT_DIR):
    raise FileNotFoundError(f"Output directory not found: {OUT_DIR}")

all_files = [f for f in os.listdir(OUT_DIR) if f.endswith("_HF1.csv")]

pred_files = []
for f in sorted(all_files):
    metrics_file = f.replace(".csv", "_metrics.csv")
    metrics_path = os.path.join(OUT_DIR, metrics_file)
    if os.path.exists(metrics_path):
        print(f"  Skipping {f} -- metrics already computed: {metrics_file}")
        continue
    pred_files.append(f)

if not pred_files:
    print(f"\nAll {len(all_files)} file(s) already have metrics computed. Nothing to do.")
    exit(0)

print(f"\nFound {len(pred_files)} new file(s) to evaluate (out of {len(all_files)} total):")
for f in pred_files:
    print(f"  {f}")

summary = []

for fname in pred_files:
    base = fname.replace("_HF1.csv", "")
    parts = base.split("_")

    prompt_name = parts[-1]
    model_and_classes = "_".join(parts[:-1])

    known_classes = ["Energy", "Safety", "Other", "Entertainment", "Health"]
    detected_classes = [c for c in known_classes if c in base]

    metrics_file = os.path.join(OUT_DIR, fname.replace(".csv", "_metrics.csv"))

    print(f"\n{'='*60}")
    print(f">>> {model_and_classes} | Prompt: {prompt_name} <<<")
    print(f"{'='*60}")

    pred_df = pd.read_csv(os.path.join(OUT_DIR, fname), encoding='utf-8')
    print(f"Loaded {len(pred_df)} predictions")

    if detected_classes:
        selected_classes = detected_classes
    else:
        selected_classes = sorted(pred_df['true_label'].dropna().unique().tolist())

    print(f"Detected classes: {selected_classes}")

    # Drop rows with NaN true_label or predicted_label
    before = len(pred_df)
    pred_df = pred_df.dropna(subset=['true_label', 'predicted_label']).reset_index(drop=True)
    if len(pred_df) < before:
        print(f"  WARNING: Dropped {before - len(pred_df)} rows with NaN labels")

    # --- Classification (EVERY row counts) ---
    cm = clf_metrics(
        pred_df['true_label'].tolist(),
        pred_df['predicted_label'].tolist(),
        selected_classes,
    )

    print(f"\n  OVERALL ACCURACY: {cm['accuracy']*100:.2f}%  ({cm['correct']}/{cm['total']})")
    print(f"  Macro Precision:  {cm['precision']:.4f}")
    print(f"  Macro Recall:     {cm['recall']:.4f}")
    print(f"  Macro F1:         {cm['macro_f1']:.4f}")
    print(f"\n  PREDICTION BREAKDOWN:")
    print(f"    Correct (right class):     {cm['valid_correct']:>4}")
    print(f"    Wrong (wrong class):       {cm['valid_wrong']:>4}")
    print(f"    Invalid (not in task):     {cm['invalid_predicted']:>4}")
    print(f"    UNCLASSIFIED:              {cm['unclassified']:>4}")

    # --- Explanation (only valid predictions with good explanations) ---
    valid = pred_df[
        (pred_df['reference_explanation'].astype(str).str.len() > 10) &
        (pred_df['generated_explanation'].astype(str).str.len() > 10)
    ].copy()

    print(f"\n  Valid pairs for explanation metrics: {len(valid)}")

    if len(valid) > 0:
        em = exp_metrics(
            valid['reference_explanation'].tolist(),
            valid['generated_explanation'].tolist(),
            device=DEVICE,
        )
    else:
        em = {
            'preservation': 0.0, 'bertscore_f1': 0.0,
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0
        }

    print(f"  Preservation={em['preservation']:.3f}  BERT={em['bertscore_f1']:.3f}  "
          f"ROUGE-1={em['rouge1']:.3f}  ROUGE-2={em['rouge2']:.3f}  "
          f"ROUGE-L={em['rougeL']:.3f}  METEOR={em['meteor']:.3f}")

    # --- Time stats ---
    total_time_min = pred_df['time_taken'].sum() / 60
    avg_time_per_req = pred_df['time_taken'].mean()

    met = {
        'file': fname,
        'model_and_classes': model_and_classes,
        'prompt': prompt_name,
        'classes': '_'.join(selected_classes),
        'accuracy':         cm['accuracy'],
        'precision':        cm['precision'],
        'recall':           cm['recall'],
        'macro_f1':         cm['macro_f1'],
        'total':            cm['total'],
        'correct':          cm['correct'],
        'wrong':            cm['wrong'],
        'valid_correct':    cm['valid_correct'],
        'valid_wrong':      cm['valid_wrong'],
        'invalid_predicted': cm['invalid_predicted'],
        'unclassified':     cm['unclassified'],
        **em,
        'avg_time_per_req': round(avg_time_per_req, 2),
        'total_samples':    len(pred_df),
        'valid_predictions': len(valid),
        'total_time_min':   round(total_time_min, 2),
    }

    pd.DataFrame([met]).to_csv(metrics_file, index=False, encoding='utf-8')
    summary.append(met)
    print(f"  Saved: {metrics_file}")

# ============================================================
# CELL 8: Final Summary
# ============================================================

if summary:
    sum_df = pd.DataFrame(summary)
    cols = ['file', 'model_and_classes', 'prompt', 'classes',
            'accuracy', 'precision', 'recall', 'macro_f1',
            'total', 'correct', 'wrong',
            'valid_correct', 'valid_wrong', 'invalid_predicted', 'unclassified',
            'preservation', 'bertscore_f1', 'rouge1', 'rouge2', 'rougeL', 'meteor',
            'avg_time_per_req', 'total_samples', 'valid_predictions', 'total_time_min']
    sum_df = sum_df[[c for c in cols if c in sum_df.columns]]

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(sum_df.to_string(index=False))

    sum_df.to_csv(os.path.join(OUT_DIR, "summary_quaternary_llama.csv"), index=False, encoding='utf-8')
    print(f"\nSummary saved: {OUT_DIR}/summary.csv")
else:
    print("\nNo new files evaluated. All metrics already computed.")
