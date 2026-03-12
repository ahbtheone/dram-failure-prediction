# DRAM Failure Prediction

This repository implements a DRAM failure prediction pipeline inspired by the ScaleDFP framework.

The project contains two experimental pipelines:

1. Single-Machine Feature Extraction
2. Distributed Feature Extraction (Collector Architecture)

Both pipelines generate datasets used to train a LightGBM model for predicting server failures.

---

## Pipeline Overview

Raw DRAM Logs
→ Daily Feature Extraction
→ Row/Burst Feature Generation
→ Dataset Merge
→ LightGBM Training
→ Online Sliding-Window Evaluation

---

## Single-Machine Pipeline

1_generate_daily_features.py
2_generate_row_burst_features.py
3_merge_dataset.py
4_train_model.py
5_online_eval.py

---

## Distributed Pipeline

1_split_logs.py
2_collector_daily_features.py
3_collector_row_burst.py
4_merge_collectors.py
5_train_distributed_model.py
run_collectors_parallel.sh
