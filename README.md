# DRAM Failure Prediction

This repository implements a DRAM failure prediction pipeline based on large-scale hardware error logs.
The system processes memory error events, generates temporal and spatial features, and trains machine learning models to predict server failures.

The project explores both a **single-machine preprocessing pipeline** and a **distributed feature generation workflow** inspired by scalable failure prediction systems such as **ScaleDFP**.

---

# Overview

Large-scale data centers continuously collect hardware telemetry logs such as DRAM error reports.
Analyzing these logs can reveal early indicators of hardware degradation and enable proactive failure prediction.

This project implements a preprocessing and training pipeline that:

* processes DRAM error logs
* extracts predictive features from error behavior
* generates labeled datasets for machine learning
* evaluates models under an online prediction setting

Two execution modes are supported:

* **Single-machine pipeline** for baseline experiments
* **Distributed preprocessing pipeline** for scalable data processing

---

# Processing Pipelines

## Single-Machine Pipeline

The single-machine pipeline processes the entire dataset on a single node and serves as the baseline workflow for feature extraction and model training.

The pipeline performs:

* Daily feature generation from raw error logs
* Row-aware and burst-aware feature extraction
* Dataset construction and labeling
* Model training and evaluation

Workflow:

```
Raw Logs
   ↓
Daily Feature Generation
   ↓
Row/Burst Feature Extraction
   ↓
Dataset Merge
   ↓
Model Training
   ↓
Online Evaluation
```

Relevant scripts:

```
single_machine/
    01_generate_daily_features.py
    02_generate_row_burst_features.py
    03_merge_dataset.py
    04_train_model.py
    05_online_eval.py
```

---

## Distributed Preprocessing Pipeline

To improve scalability when processing large volumes of telemetry data, the repository includes a distributed preprocessing workflow inspired by the **ScaleDFP architecture**.

In this setup, the raw logs are partitioned and processed by multiple collectors.

```
Log Partitions
     ↓
Data Collectors
     ↓
Feature Aggregation
     ↓
Merged Dataset
     ↓
Training Pipeline
```

Each collector processes a subset of servers and performs **near-data preprocessing** to generate local features.
Collector outputs are merged into a unified dataset used for model training.

Distributed scripts:

```
distributed/
    01_split_logs.py
    02_collector_daily_features.py
    03_collector_row_burst.py
    04_merge_collectors.py
    05_train_distributed_model.py
```

---

# Feature Engineering

The system extracts several types of predictive features from DRAM error logs.

## Temporal Features

* `ce_count_past`
* `mean_inter_error_time_past`
* `ce_last_1d`
* `ce_last_3d`

## Spatial Features

* `unique_banks_past`
* `unique_rows_past`
* `row_max_count_past`
* `row_entropy_past`

## Burst Features

* `burst_count_1h`
* `burst_count_1d`

These features capture both the **temporal evolution of errors** and the **locality of error occurrences**, which are strong indicators of hardware failure.

---

# Repository Structure

```
dram-failure-prediction
│
├── single_machine
│   ├── 01_generate_daily_features.py
│   ├── 02_generate_row_burst_features.py
│   ├── 03_merge_dataset.py
│   ├── 04_train_model.py
│   └── 05_online_eval.py
│
├── distributed
│   ├── 01_split_logs.py
│   ├── 02_collector_daily_features.py
│   ├── 03_collector_row_burst.py
│   ├── 04_merge_collectors.py
│   ├── 05_train_distributed_model.py
│   └── run_collectors_parallel.sh
│
├── analysis
│   └── ce_baseline_analysis.py
│
├── archive
│   └── experimental scripts used during development
│
└── README.md
```

---

# Requirements

Python ≥ 3.8

Required libraries:

* pandas
* numpy
* lightgbm
* scikit-learn

Install dependencies:

```
pip install -r requirements.txt
```

---

# Dataset

The pipeline expects DRAM error logs with fields such as:

* `sid`
* `memoryid`
* `rankid`
* `bankid`
* `row`
* `error_time`

Failure labels are obtained from **trouble ticket logs**.

Example dataset files:

```
data/raw/mcelog.csv
data/raw/trouble_tickets.csv
```

Due to dataset size constraints, the raw datasets are **not included** in this repository.

---

# Running the Pipeline

## Single-machine pipeline

```
python single_machine/01_generate_daily_features.py
python single_machine/02_generate_row_burst_features.py
python single_machine/03_merge_dataset.py
python single_machine/04_train_model.py
python single_machine/05_online_eval.py
```

## Distributed preprocessing

```
python distributed/01_split_logs.py
bash distributed/run_collectors_parallel.sh
python distributed/04_merge_collectors.py
python distributed/05_train_distributed_model.py
```

---

# Experimental Results

Using the generated feature set and a **LightGBM classifier**, the system achieves:

* Precision ≈ 0.27
* Recall ≈ 0.63
* F1 Score ≈ 0.38

under an **online sliding-window evaluation setting**.

---

# Relation to ScaleDFP

The distributed preprocessing design in this repository is inspired by:

**ScaleDFP: Scaling Disk Failure Prediction via Multi-Source Stream Mining**

This implementation focuses primarily on:

* DRAM error log preprocessing
* feature engineering
* simplified distributed feature generation

rather than the full distributed learning infrastructure described in the original system.

---

# License

Apache License 2.0
