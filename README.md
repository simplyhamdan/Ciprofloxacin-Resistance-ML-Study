# Ciprofloxacin Resistance ML Study

A comparative machine learning study investigating how different machine learning algorithms classify ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes using genome-derived features.

## Research Question

How do different machine learning algorithms compare in classifying ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes using genome-level features?

## Project Overview

This project uses publicly available antimicrobial resistance phenotype data together with genome-derived numerical features to compare the performance of multiple machine learning algorithms.

The project is designed as a **comparative machine learning study**, rather than a clinical prediction tool.

### Dataset

The dataset contains:

* 9,540 resistant genomes
* 9,926 susceptible genomes
* 19,466 genomes in total
* Ciprofloxacin resistance phenotype labels
* Genome-derived numerical features

### Genome-derived features

The machine learning models use:

* Genome length
* GC content
* CDS count
* rRNA count
* tRNA count
* Contig count
* Contig N50
* Contig L50
* Hypothetical CDS count
* Plasmid count
* CheckM completeness
* CheckM contamination

## Machine Learning Models

Eight classification algorithms are compared:

1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Support Vector Machine
5. K-Nearest Neighbors
6. Gradient Boosting
7. Gaussian Naive Bayes
8. XGBoost

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion matrices
* 5-fold cross-validation
* Training time
* Prediction time

The project also provides feature-analysis methods for models that expose feature importance or coefficients.

## Visualizations

The program can generate:

* Resistance distribution
* Genome length distributions
* GC-content distributions
* Feature correlation heatmaps
* Resistant vs susceptible feature comparisons
* Confusion matrices
* Model comparison graphs
* Feature importance plots
* Logistic regression coefficient plots

## Data Processing

The dataset preparation includes:

* Removing conflicting genome IDs appearing in both resistance classes
* Removing duplicate genome records
* Retrieving genome-derived features
* Combining resistance labels with genome features
* Checking missing values and data types
* Median imputation for missing feature values
* Train/test splitting with stratification

## Project Structure

```text
Ciprofloxacin-Resistance-ML-Study/
│
├── data/
│   ├── ciprofloxacin_resistant.csv
│   ├── ciprofloxacin_susceptible.csv
│   ├── genome_features.csv
│   └── feature_data.csv
│
├── main.py
├── README.md
└── .gitignore
```

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* BV-BRC/PATRIC antimicrobial resistance data

## Important Limitations

This study should not be interpreted as a clinical antimicrobial resistance prediction system.

The models use **general genome-derived features**, rather than directly analysing resistance genes, mutations, or complete genome sequences.

The dataset also contains multiple bacterial species. Therefore, some model performance may reflect species-associated genome characteristics rather than ciprofloxacin resistance mechanisms themselves.

The results should therefore be interpreted as a **comparative machine learning experiment** on the selected dataset and features.

## Purpose

This project was developed as a bioinformatics and machine learning study to explore:

* Biological data preprocessing
* Genome-derived feature extraction
* Supervised machine learning
* Model comparison
* Model evaluation
* Data visualization
* Interpretation of machine learning results

## Author

Hamdan Sajith
