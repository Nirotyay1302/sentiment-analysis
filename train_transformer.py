"""
Fine-tune a Hugging Face transformer for 3-class sentiment.
Usage examples:
  # use tweet_eval
  python train_transformer.py
  # use your CSV with columns 'text' and 'label' (labels 0/1/2 or strings 'Negative'...)
  python train_transformer.py --data mydata.csv --text-col comment --label-col sentiment --model distilbert-base-uncased --epochs 3 --batch 16
"""
import os
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

def load_csv_as_hf(data_path, text_col='text', label_col='label', test_size=0.1, seed=42):
    df = pd.read_csv(data_path)
    if text_col not in df.columns:
        # fallback: first string column
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    if label_col not in df.columns:
        raise ValueError("Label column not found; pass --label-col")
    texts = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].astype(str).str.lower().tolist()
    def map_label(x):
        x = str(x).strip().lower()
        if x in LABEL_MAP:
            return LABEL_MAP[x]
        try:
            iv = int(x)
            return iv
        except:
            return 1
    labels = [map_label(l) for l in labels_raw]
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)
    return DatasetDict({
        "train": Dataset.from_dict({"text": tr_texts, "label": tr_labels}),
        "validation": Dataset.from_dict({"text": val_texts, "label": val_labels})
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV path; if omitted uses tweet_eval", default=None)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--output", default="transformer_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if args.data:
        ds = load_csv_as_hf(args.data, text_col=args.text_col, label_col=args.label_col)
    else:
        ds = load_dataset("tweet_eval", "sentiment")
        # merge train+validation to follow our train/val split logic
        all_texts = ds["train"]["text"] + ds["validation"]["text"]
        all_labels = ds["train"]["label"] + ds["validation"]["label"]
        tr_texts, val_texts, tr_labels, val_labels = train_test_split(all_texts, all_labels, test_size=0.1, random_state=42, stratify=all_labels)
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": tr_texts, "label": tr_labels}),
            "validation": Dataset.from_dict({"text": val_texts, "label": val_labels}),
            "test": Dataset.from_dict({"text": ds['test']['text'], "label": ds['test']['label']})
        })

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, ignore_mismatched_sizes=True)

    if EVALUATE_AVAILABLE:
        metric_acc = evaluate.load("accuracy")
        metric_f1 = evaluate.load("f1")
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
            f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
            return {"accuracy": acc, "f1": f1}
    else:
        # Fallback using sklearn
        from sklearn.metrics import accuracy_score, f1_score
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
            return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        push_to_hub=False,
        fp16=True if (os.environ.get("ACCELERATE_USE_FP16") or False) else False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    results = trainer.evaluate(ds["validation"])
    print("Validation results:", results)
    if "test" in ds:
        print("Test evaluation...")
        print(trainer.evaluate(ds["test"]))
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()