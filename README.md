# Ciprofloxacin Resistance ML Study

A comparative machine learning study investigating whether genome-level features can be used to classify bacterial genomes as **ciprofloxacin-resistant** or **ciprofloxacin-susceptible**.

The project compares multiple machine learning algorithms and evaluates their performance using several classification metrics, cross-validation, feature analysis, visualizations, and computational-time measurements.

---

## Research Question

> **How do different machine learning algorithms compare in classifying ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes using genome-level features?**

---

## Project Overview

Antimicrobial resistance (AMR) is an important challenge in microbiology and infectious disease research.

This project investigates whether broad genomic characteristics can provide useful information for distinguishing bacterial genomes associated with ciprofloxacin resistance.

Rather than focusing on a single machine learning algorithm, the project performs a **comparative study of eight classification models**.

The workflow includes:

1. Collecting ciprofloxacin resistance phenotype data
2. Cleaning and preparing the phenotype dataset
3. Retrieving genome-level information
4. Extracting numerical genomic features
5. Creating resistant/susceptible labels
6. Training multiple machine learning models
7. Evaluating model performance
8. Performing cross-validation
9. Comparing computational time
10. Analyzing feature importance
11. Visualizing the dataset and model results

---

## Dataset

The phenotype data was obtained from the **Bacterial and Viral Bioinformatics Resource Center (BV-BRC)**.

The study focuses on:

**Antibiotic:** Ciprofloxacin

**Classes:**

* Resistant
* Susceptible

### Current Dataset

After cleaning:

| Category    |    Genomes |
| ----------- | ---------: |
| Resistant   |      9,540 |
| Susceptible |      9,926 |
| **Total**   | **19,466** |

The dataset was cleaned by:

* Removing genomes appearing in both phenotype groups
* Removing duplicate genome records
* Matching phenotype information with genome-level information
* Handling missing numerical feature values

---

## Genome Features

The models currently use the following genome-level features:

* Genome length
* GC content
* CDS
* rRNA
* tRNA
* Contigs
* Contig N50
* Contig L50
* Hypothetical CDS
* Plasmids
* CheckM completeness
* CheckM contamination

The `genome_id` is used to connect datasets but is not used as a predictive feature.

### Target Variable

The resistance phenotype is encoded as:

```text
0 = Susceptible
1 = Resistant
```

---

## Machine Learning Models

Eight classification algorithms are currently compared:

### 1. Logistic Regression

A linear classification model used as a baseline approach.

### 2. Random Forest

An ensemble of decision trees used to capture nonlinear relationships between genomic features and resistance.

### 3. Decision Tree

A tree-based classification model that provides an interpretable decision structure.

### 4. Support Vector Machine

A classification algorithm used to separate the two phenotype groups in feature space.

### 5. K-Nearest Neighbors

A distance-based classification algorithm that predicts the class based on nearby observations.

### 6. Gradient Boosting

An ensemble method that builds sequential decision trees to improve classification performance.

### 7. Gaussian Naive Bayes

A probabilistic classification method based on Bayes' theorem.

### 8. XGBoost

A gradient-boosted tree algorithm included for comparison with the other classification approaches.

---

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Cross-Validation

A **5-fold cross-validation** analysis is included to examine model performance across multiple training subsets.

For each model, the following are recorded:

* Fold accuracy scores
* Mean accuracy
* Standard deviation

### Computational Performance

Training and prediction times are also measured for each model.

This allows the study to compare not only predictive performance but also computational requirements.

---

## Feature Analysis

Feature analysis is included to investigate which genomic variables contribute most strongly to model predictions.

The project currently supports:

* Random Forest feature importance
* Decision Tree feature importance
* Gradient Boosting feature importance
* XGBoost feature importance
* Logistic Regression coefficients

These analyses can help identify genomic characteristics associated with the model's classification decisions.

---

## Visualizations

The program includes several visualization options.

### Dataset Visualizations

* Resistance distribution
* Genome length distribution
* GC content distribution
* Feature correlation heatmap
* Resistant vs susceptible feature distributions

### Model Visualizations

* Confusion matrices
* Accuracy comparison
* Precision comparison
* Recall comparison
* F1-score comparison
* ROC-AUC comparison

---

## Project Structure

```text
Ciprofloxacin-Resistance-ML-Study/
│
├── data/
│   ├── ciprofloxacin_resistant.csv
│   ├── ciprofloxacin_susceptible.csv
│   ├── feature_data.csv
│   └── genome_features.csv
│
├── main.py
├── README.md
└── Research_Study.md
```

> The exact contents of the `data/` directory may vary depending on whether genome data has already been downloaded locally.

---

## Program Features

The program provides a menu-driven interface with the following main sections:

```text
CIPROFLOXACIN RESISTANCE ML STUDY

[1] Machine Learning Models
[2] Visualize Dataset
[3] Feature Analysis
[4] View Internal Messages
[5] Exit
```

### Machine Learning Models

The model menu allows individual inspection of:

* Accuracy
* ROC-AUC
* Classification report
* Confusion matrix

It also provides:

* Model comparison
* 5-fold cross-validation
* Model training/prediction time comparison

### Visualization

The visualization menu provides access to dataset plots, feature comparisons, confusion matrices, and model comparison graphs.

### Feature Analysis

The feature analysis menu provides model-specific feature importance and coefficient analysis.

### Internal Messages

Internal processing information is stored separately so that dataset preparation and API-related messages do not clutter the main program interface.

---

## Data Processing Workflow

```text
Phenotype Data
      │
      ▼
Remove Conflicting Genomes
      │
      ▼
Remove Duplicate Genomes
      │
      ▼
Retrieve Genome Information
      │
      ▼
Select Genome-Level Features
      │
      ▼
Create Resistance Labels
      │
      ▼
Merge Features + Labels
      │
      ▼
Handle Missing Values
      │
      ▼
Train/Test Split
      │
      ▼
Train Machine Learning Models
      │
      ├── Logistic Regression
      ├── Random Forest
      ├── Decision Tree
      ├── SVM
      ├── KNN
      ├── Gradient Boosting
      ├── Gaussian Naive Bayes
      └── XGBoost
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1 Score
      ├── ROC-AUC
      └── Confusion Matrix
      │
      ▼
Cross-Validation
      │
      ▼
Feature Analysis
      │
      ▼
Model Comparison
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* pandas
* requests
* matplotlib
* seaborn
* scikit-learn
* XGBoost

### Data Source

* BV-BRC

---

## Limitations

This project currently uses broad genome-level characteristics rather than specific resistance genes, mutations, or sequence-derived resistance markers.

Therefore, the models may identify associations between genomic composition and resistance classification without identifying the biological mechanism responsible for ciprofloxacin resistance.

The dataset also contains multiple bacterial species. Differences in genome structure and composition between species may influence the classification results.

The dataset is based on publicly available records and may not represent all bacterial populations or geographic regions.

Model performance on this dataset should not automatically be interpreted as performance on an independent external dataset.

Phenotype classifications originate from the underlying public database and may reflect differences in testing conditions, standards, and measurement procedures.

---

## Research Documentation

A more detailed description of the study is available in:

**`Research_Study.md`**

The research document contains:

* Background
* Research question
* Objectives
* Dataset construction
* Feature descriptions
* Data preprocessing
* Machine learning methodology
* Evaluation methodology
* Results tables
* Feature analysis
* Discussion
* Limitations
* Future work
* Reproducibility information
* References

The results and conclusions in the research document will be updated using the final experimental outputs.

---

## Purpose

This project was developed as a **bioinformatics and machine learning study** to explore the intersection of bacterial genomics, antimicrobial resistance, and predictive modeling.

The main goal is not simply to build a resistance classifier, but to investigate how different machine learning approaches behave when applied to the same genomic dataset.

---

## Author

**Hamdan Sajith**

Bioinformatics / Machine Learning Project

---
