"""LightGBM benchmark on the Credit Card Fraud Detection dataset (Lab 16, Phần 4.4).

Usage (on the compute node):
    python3 benchmark.py                       # reads ~/ml-benchmark/creditcard.csv
    python3 benchmark.py --data path/to/creditcard.csv --out benchmark_result.json
"""

import argparse
import json
import os
import platform
import statistics
import time
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42

# Newer LightGBM prefers eval_X/eval_y; eval_set still works and keeps older versions compatible
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM fraud-detection benchmark")
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_time = time.perf_counter() - t0

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train 70% / validation 10% (early stopping) / test 20%, stratified because fraud is ~0.17%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=SEED
    )

    # 2. Train
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        # Only ~0.17% positives: without these, leaves holding a handful of frauds get extreme
        # values, validation loss diverges after the first tree and early stopping halts at 1
        min_child_weight=1.0,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    train_time = time.perf_counter() - t0

    # 3. Evaluate on the held-out test set
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
    }

    # 4. Inference latency (1 row) — median of 100 runs after warm-up
    one_row = X_test.iloc[[0]]
    for _ in range(10):
        model.predict_proba(one_row)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        latencies.append(time.perf_counter() - t0)
    latency_ms = statistics.median(latencies) * 1000

    # 5. Inference throughput (1000 rows) — median of 20 runs
    batch = X_test.iloc[:1000]
    batch_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict_proba(batch)
        batch_times.append(time.perf_counter() - t0)
    batch_ms = statistics.median(batch_times) * 1000
    throughput = len(batch) / (batch_ms / 1000)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "cpu_count": os.cpu_count(),
        },
        "dataset": {
            "path": args.data,
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "fraud_ratio": float(y.mean()),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "load_time_s": round(load_time, 4),
        "train_time_s": round(train_time, 4),
        "best_iteration": int(model.best_iteration_),
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "inference": {
            "latency_1_row_ms": round(latency_ms, 4),
            "batch_1000_rows_ms": round(batch_ms, 4),
            "throughput_rows_per_s": round(throughput, 1),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 52)
    print(" LightGBM Credit Card Fraud Benchmark")
    print("=" * 52)
    print(f" Rows / features          : {len(df):,} / {X.shape[1]}")
    print(f" Data load time           : {load_time:.3f} s")
    print(f" Training time            : {train_time:.3f} s")
    print(f" Best iteration           : {model.best_iteration_}")
    print(f" AUC-ROC                  : {metrics['auc_roc']:.4f}")
    print(f" Accuracy                 : {metrics['accuracy']:.4f}")
    print(f" F1-Score                 : {metrics['f1_score']:.4f}")
    print(f" Precision                : {metrics['precision']:.4f}")
    print(f" Recall                   : {metrics['recall']:.4f}")
    print(f" Inference latency (1 row): {latency_ms:.3f} ms")
    print(f" Throughput (1000 rows)   : {batch_ms:.3f} ms -> {throughput:,.0f} rows/s")
    print("=" * 52)
    print(f" Saved results to {args.out}")


if __name__ == "__main__":
    main()
