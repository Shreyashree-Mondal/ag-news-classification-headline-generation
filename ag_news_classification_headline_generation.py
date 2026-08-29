# ================================
# PEGASUS Fine-tuning for AG News Headline Generation
# Team Member #3: Shreyashree Mondal
# ================================

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from evaluate import load
import warnings
import re
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import html

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # keep original attempt; safe
    nltk.download('stopwords', quiet=True)
except:
    pass

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

print("="*80)
print("CUDA ENVIRONMENT SETUP")
print("="*80)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("="*80)
print("LOADING AG NEWS DATASET")
print("="*80)
dataset = load_dataset("fancyzhx/ag_news")
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

print("Dataset: fancyzhx/ag_news\n")

# Raw sizes
train_raw = dataset['train']
test_raw = dataset['test']
print(f"Training samples (raw): {len(train_raw):,}")
print(f"Test samples: {len(test_raw):,}")
print(f"Columns: {list(train_raw.features.keys())}\n")

# ================================
# PREPROCESSING: PSEUDO-HEADLINE EXTRACTION
# ================================

def clean_headline(text):
    """Clean only the headline (pseudo-label). Convert HTML entities, preserve meaning, avoid aggressive stripping."""
    text = html.unescape(text)  # convert entities like &#39; → '
    text = re.sub(r"http\S+", " ", text)  # remove URLs
    text = re.sub(r"<.*?>", " ", text)    # NEW ADDITION: remove HTML tags
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)  # NEW ADDITION: remove control characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_headline_pairs(examples):
    descriptions, headlines = [], []
    for text in examples['text']:
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: sentences = [text]
        headline = clean_headline(sentences[0][:120])
        description = ' '.join(sentences[1:]) if len(sentences) > 1 else sentences[0]
        if not description or len(description.strip()) < 5:
            description = text
        descriptions.append(description)
        headlines.append(headline)
    return {'description': descriptions, 'headline': headlines, 'label': examples['label']}

# Split train into train + validation
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset, valid_dataset, test_dataset = split_dataset['train'], split_dataset['test'], dataset['test']

train_dataset = train_dataset.map(create_headline_pairs, batched=True)
valid_dataset = valid_dataset.map(create_headline_pairs, batched=True)
test_dataset  = test_dataset.map(create_headline_pairs, batched=True)

print("="*80)
print("PREPROCESSING: PSEUDO-HEADLINE EXTRACTION")
print("="*80)
print("""
Approach: Split text into pseudo-headline and description
  - First sentence (max 120 chars) -> Target (pseudo-headline) [cleaned minimally]
  - Remaining text -> Input (description) [unaltered]
  - ROUGE/BERTScore compare: generated vs pseudo-headline
  - DO NOT uppercase during training; uppercase only in printed samples
""")

# ================================
# CONFIG: Always use full dataset unless overridden
# ================================
USE_FULL_TRAIN = True
USE_FULL_VALID = True
USE_FULL_TEST  = True

TRAINING_SAMPLES   = 50
VALIDATION_SAMPLES = 10
TEST_SAMPLES       = 10

if not USE_FULL_TRAIN:
    train_dataset = train_dataset.shuffle(seed=42).select(range(TRAINING_SAMPLES))
if not USE_FULL_VALID:
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(VALIDATION_SAMPLES))
if not USE_FULL_TEST:
    test_dataset = test_dataset.shuffle(seed=42).select(range(TEST_SAMPLES))

print(f"Using split: Train={len(train_dataset):,}, Validation={len(valid_dataset):,}, Test={len(test_dataset):,}")
print()

# ================================
# MODEL SETUP
# ================================

model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

print("="*80)
print("MODEL CONFIG SUMMARY")
print("="*80)
print(model.config)

# NEW ADDITION: Print model architecture summary
print("="*80)
print("MODEL ARCHITECTURE SUMMARY")
print("="*80)
print(model)  # prints full layer structure

# ================================
# TOKENIZATION
# ================================

def preprocess_function(examples):
    inputs, targets = examples['description'], examples['headline']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True,
                                    remove_columns=['description','headline','text'])
tokenized_valid = valid_dataset.map(preprocess_function, batched=True,
                                    remove_columns=['description','headline','text'])
tokenized_test  = test_dataset.map(preprocess_function, batched=True,
                                    remove_columns=['description','headline','text'])

tokenized_train.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
tokenized_valid.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
tokenized_test.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# ================================
# TRAINING CONFIG
# ================================

training_args = Seq2SeqTrainingArguments(
    output_dir="./pegasus-agnews",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    weight_decay=0.01,
    warmup_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    report_to="none",
    generation_max_length=64,
    generation_num_beams=6,
)

print("="*80)
print("TRAINING CONFIG SUMMARY")
print("="*80)
print(training_args)

# NEW ADDITION: Explicit decoding settings printout
print("="*80)
print("DECODING SETTINGS SUMMARY")
print("="*80)
print("Beam search: num_beams=6, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True")
print("Additional strategies: greedy, temperature=0.77, top-k=50, top-p=0.9")

# ================================
# METRICS
# ================================

rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bertscore_result = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    return {
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "bertscore_f1": float(np.mean(bertscore_result["f1"]))
    }

# ================================
# TRAINING
# ================================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train()

print("="*80)
print("BEST MODEL CHECKPOINT INFO")
print("="*80)
print(f"Best model loaded from: {trainer.state.best_model_checkpoint}")
print(f"Best validation ROUGE-1 (F1): {trainer.state.best_metric:.4f}")

# ================================
# VALIDATION EVALUATION
# ================================

eval_results = trainer.evaluate()
print("\nEVALUATION RESULTS:")
print("="*80)
print(f"  ROUGE-1 (F1): {eval_results['eval_rouge1']:.4f}")
print(f"  ROUGE-2 (F1): {eval_results['eval_rouge2']:.4f}")
print(f"  ROUGE-L (F1): {eval_results['eval_rougeL']:.4f}")
print(f"  BERTScore (F1): {eval_results['eval_bertscore_f1']:.4f}")

print("="*80)
print("TRAIN VS VALIDATION LOSS CURVE")
print("="*80)
log_hist = trainer.state.log_history
train_losses = [x['loss'] for x in log_hist if 'loss' in x]
eval_losses = [x['eval_loss'] for x in log_hist if 'eval_loss' in x]

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(eval_losses, label="Validation Loss")
plt.xlabel("Logged steps / epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

print("="*80)
print("EPOCH-ALIGNED TRAIN VS VALIDATION LOSS")
print("="*80)
train_loss_by_epoch = defaultdict(list)
eval_loss_by_epoch = {}
for rec in trainer.state.log_history:
    if "loss" in rec and "epoch" in rec:
        train_loss_by_epoch[rec["epoch"]].append(rec["loss"])
    if "eval_loss" in rec and "epoch" in rec:
        eval_loss_by_epoch[rec["epoch"]] = rec["eval_loss"]

epoch_train_losses = sorted([(ep, np.mean(vals)) for ep, vals in train_loss_by_epoch.items()], key=lambda x: x[0])
epochs_t = [ep for ep, _ in epoch_train_losses]
train_means = [v for _, v in epoch_train_losses]
epochs_e = sorted(eval_loss_by_epoch.keys())
eval_means = [eval_loss_by_epoch[ep] for ep in epochs_e]

plt.figure(figsize=(8,5))
plt.plot(epochs_t, train_means, 'o-', label="Train Loss (epoch avg)", color="steelblue")
plt.plot(epochs_e, eval_means, 'o-', label="Validation Loss (epoch)", color="darkorange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch-aligned Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

print("="*80)
print("SAMPLE PREDICTIONS (BEAM SEARCH ONLY)")
print("="*80)
for i in range(5):
    description = valid_dataset[i]['description']
    true_headline = valid_dataset[i]['headline']
    label = label_map[valid_dataset[i]['label']]
    inputs = tokenizer(description, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=64,
            num_beams=6,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    generated_headline = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"{'='*80}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*80}")
    print(f"Category:            {label}")
    print(f"Gold Headline:       {true_headline.upper()}")
    print(f"Generated Headline:  {generated_headline.upper()}")
    print(f"Description:         {description[:200]}...")
    print()

print("="*80)
print("CATEGORY-WISE EVALUATION (VALIDATION)")
print("="*80)
rouge_scores = defaultdict(list)
bert_scores = defaultdict(list)
all_rouge, all_bert = [], []

for i in range(len(valid_dataset)):
    desc = valid_dataset[i]['description']
    gold = valid_dataset[i]['headline']
    lab_name = label_map[valid_dataset[i]['label']]
    inputs = tokenizer(desc, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        pred_ids = model.generate(inputs['input_ids'], max_length=64, num_beams=6)
    pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    rr = rouge_metric.compute(predictions=[pred], references=[gold], use_stemmer=True)
    rouge_l_f1 = float(rr['rougeL'])
    rouge_scores[lab_name].append(rouge_l_f1)
    all_rouge.append(rouge_l_f1)
    br = bertscore_metric.compute(predictions=[pred], references=[gold], lang="en")
    bert_f1 = float(br['f1'][0])
    bert_scores[lab_name].append(bert_f1)
    all_bert.append(bert_f1)

rows = []
for cat in label_map.values():
    rows.append({
        "Category": cat,
        "ROUGE-L (F1)": np.mean(rouge_scores[cat]) if rouge_scores[cat] else np.nan,
        "BERTScore (F1)": np.mean(bert_scores[cat]) if bert_scores[cat] else np.nan
    })
rows.append({"Category": "Overall", "ROUGE-L (F1)": np.mean(all_rouge), "BERTScore (F1)": np.mean(all_bert)})
cat_df = pd.DataFrame(rows)
print("\nCATEGORY-WISE + OVERALL SCORES")
print(cat_df.to_string(index=False, float_format="%.4f"))

plt.figure(figsize=(8,5))
cats = [r["Category"] for r in rows[:-1]]
rouge_vals = [r["ROUGE-L (F1)"] for r in rows[:-1]]
bert_vals = [r["BERTScore (F1)"] for r in rows[:-1]]
x = np.arange(len(cats))
w = 0.35
plt.bar(x - w/2, rouge_vals, w, label="ROUGE-L (F1)", color="cornflowerblue")
plt.bar(x + w/2, bert_vals, w, label="BERTScore (F1)", color="mediumseagreen")
plt.xticks(x, cats)
plt.ylabel("Score")
plt.title("Validation scores by category")
plt.legend()
plt.tight_layout()
plt.show()

print("="*80)
print("PROJECT SUMMARY")
print("="*80)
print(f"""
Training Configuration:
  - Train samples: {len(train_dataset):,}
  - Validation samples: {len(valid_dataset):,}
  - Test samples: {len(test_dataset):,}
  - Epochs: {training_args.num_train_epochs}
  - Batch size: {training_args.per_device_train_batch_size}
  - Gradient accumulation: {training_args.gradient_accumulation_steps}
  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}
  - Learning rate: {training_args.learning_rate}
  - Warmup steps: {training_args.warmup_steps}
  - FP16: {torch.cuda.is_available()}

Results (Validation):
  - ROUGE-1 (F1): {eval_results['eval_rouge1']:.4f}
  - ROUGE-2 (F1): {eval_results['eval_rouge2']:.4f}
  - ROUGE-L (F1): {eval_results['eval_rougeL']:.4f}
  - BERTScore (F1): {eval_results['eval_bertscore_f1']:.4f}
  - Training time: {train_result.metrics['train_runtime']/60:.2f} minutes

Decoding Settings Used:
  - Beam search only (num_beams=6, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
""")

print("="*80)
print("FINAL EVALUATION (ON TEST SET)")
print("="*80)
test_results = trainer.evaluate(eval_dataset=tokenized_test)
print(f"ROUGE-1 (F1): {test_results['eval_rouge1']:.4f}")
print(f"ROUGE-2 (F1): {test_results['eval_rouge2']:.4f}")
print(f"ROUGE-L (F1): {test_results['eval_rougeL']:.4f}")
print(f"BERTScore (F1): {test_results['eval_bertscore_f1']:.4f}")

# ================================
# DECODING STRATEGY COMPARISON
# ================================

STRATEGIES = {
    "greedy": {"do_sample": False},
    "temperature_0.77": {"do_sample": True, "temperature": 0.77},
    "top_k_50": {"do_sample": True, "top_k": 50},
    "top_p_0.9": {"do_sample": True, "top_p": 0.9}
}

def generate_with_strategy(input_ids, strategy_params, max_len):
    with torch.no_grad():
        return model.generate(input_ids, max_length=max_len, num_beams=1, **strategy_params)

def evaluate_split_per_class(split_dataset, split_name, enc_max, dec_max, strategies):
    rows = []
    for strat_name, params in strategies.items():
        rouge_scores, bert_scores = defaultdict(list), defaultdict(list)
        all_rouge, all_bert = [], []
        for i in range(len(split_dataset)):
            desc, gold = split_dataset[i]['description'], split_dataset[i]['headline']
            lab_name = label_map[split_dataset[i]['label']]
            inputs = tokenizer(
                desc,
                return_tensors="pt",
                truncation=True,
                max_length=enc_max
            ).to(device)

            pred_ids = generate_with_strategy(inputs['input_ids'], params, max_len=dec_max)
            pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

            rr = rouge_metric.compute(predictions=[pred], references=[gold], use_stemmer=True)
            rouge_l_f1 = float(rr['rougeL'])
            rouge_scores[lab_name].append(rouge_l_f1)
            all_rouge.append(rouge_l_f1)

            br = bertscore_metric.compute(predictions=[pred], references=[gold], lang="en")
            bert_f1 = float(br['f1'][0])
            bert_scores[lab_name].append(bert_f1)
            all_bert.append(bert_f1)

        # Per-class rows
        for cat in label_map.values():
            rows.append({
                "Split": split_name,
                "Strategy": strat_name,
                "Category": cat,
                "ROUGE-L (F1)": np.mean(rouge_scores[cat]) if rouge_scores[cat] else np.nan,
                "BERTScore (F1)": np.mean(bert_scores[cat]) if bert_scores[cat] else np.nan
            })
        # Overall summary row
        rows.append({
            "Split": split_name,
            "Strategy": strat_name,
            "Category": "Overall",
            "ROUGE-L (F1)": np.mean(all_rouge),
            "BERTScore (F1)": np.mean(all_bert)
        })

    return pd.DataFrame(rows)

print("="*80)
print("DECODING STRATEGY COMPARISON (VALIDATION)")
print("="*80)
val_strat_df = evaluate_split_per_class(
    valid_dataset,
    split_name="Validation",
    enc_max=128,
    dec_max=64,
    strategies=STRATEGIES
)
print(val_strat_df.to_string(index=False, float_format="%.4f"))

print("="*80)
print("DECODING STRATEGY COMPARISON (TEST)")
print("="*80)
test_strat_df = evaluate_split_per_class(
    test_dataset,
    split_name="Test",
    enc_max=128,
    dec_max=64,
    strategies=STRATEGIES
)
print(test_strat_df.to_string(index=False, float_format="%.4f"))

# ================================
# VISUALIZATION: Strategy bars
# ================================

def plot_strategy_bars(df, split_name):
    # Remove overall row for plotting
    df_plot = df[df["Category"] != "Overall"]
    # Melt into long format for grouped bars
    df_melt = df_plot.melt(
        id_vars=["Strategy", "Category"],
        value_vars=["ROUGE-L (F1)", "BERTScore (F1)"],
        var_name="Metric",
        value_name="Score"
    )
    plt.figure(figsize=(10,5))
    ax = sns.barplot(
        data=df_melt,
        x="Category",
        y="Score",
        hue="Metric",
        palette=["cornflowerblue","mediumseagreen"],
        ci=None,
        width=0.30
    )
    plt.title(f"{split_name} Decoding Strategy Comparison")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    # Annotate bars (skip 0.0 and NaN)
    for p in ax.patches:
        height = p.get_height()
        if height > 0 and not np.isnan(height):
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width()/2., height),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black", xytext=(0,3),
                textcoords="offset points"
            )
    plt.tight_layout()
    plt.show()

    # Separate charts per strategy
    for strat in df_plot["Strategy"].unique():
        strat_df = df_plot[df_plot["Strategy"] == strat]
        strat_melt = strat_df.melt(
            id_vars=["Category"],
            value_vars=["ROUGE-L (F1)", "BERTScore (F1)"],
            var_name="Metric",
            value_name="Score"
        )
        plt.figure(figsize=(10,5))
        ax = sns.barplot(
            data=strat_melt,
            x="Category",
            y="Score",
            hue="Metric",
            palette=["cornflowerblue","mediumseagreen"],
            ci=None,
            width=0.30
        )
        plt.title(f"{split_name} – Strategy: {strat}")
        plt.ylabel("Score")
        plt.legend(title="Metric")

        # Annotate each bar with its numeric score
        for p in ax.patches:
            height = p.get_height()
            if height > 0 and not np.isnan(height):
                ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width()/2., height),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black", xytext=(0,3),
                textcoords="offset points"
            )

        # NEW ADDITION: highlight the highest score with a red star
        max_val = max([p.get_height() for p in ax.patches if p.get_height() > 0])
        for p in ax.patches:
            if abs(p.get_height() - max_val) < 1e-6:
                ax.plot(p.get_x() + p.get_width()/2., max_val,
                    marker="*", color="red", markersize=12)

        plt.tight_layout()
        plt.show()


print("="*80)
print("READY FOR TEAM COMPARISON: VISUALIZATIONS")
print("="*80)
plot_strategy_bars(val_strat_df, "Validation")
plot_strategy_bars(test_strat_df, "Test")