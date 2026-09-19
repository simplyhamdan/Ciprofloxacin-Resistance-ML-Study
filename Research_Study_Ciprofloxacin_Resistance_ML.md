# Ciprofloxacin Resistance ML Study

## Abstract

Antimicrobial resistance (AMR) is an important challenge in clinical microbiology and infectious disease management. Computational methods can help investigate whether genomic characteristics are associated with antimicrobial resistance phenotypes.

This study investigates how different machine learning algorithms compare in classifying bacterial genomes as ciprofloxacin-resistant or ciprofloxacin-susceptible using genome-level features obtained from the Bacterial and Viral Bioinformatics Resource Center (BV-BRC).

The study evaluates eight machine learning algorithms: Logistic Regression, Random Forest, Decision Tree, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, Gaussian Naive Bayes, and XGBoost. Model performance is assessed using accuracy, precision, recall, F1-score, ROC-AUC, 5-fold cross-validation, and computational time.

Final quantitative results will be added after the final model evaluation is completed.

---

## 1. Introduction

### 1.1 Background

Antimicrobial resistance occurs when microorganisms develop the ability to survive exposure to antimicrobial drugs that would normally inhibit or kill them. It is an important challenge because resistant infections can reduce treatment effectiveness and increase the complexity of clinical management.

Ciprofloxacin is a fluoroquinolone antibiotic used against a range of bacterial infections. Resistance to ciprofloxacin can arise through several mechanisms, including genetic changes affecting antibiotic targets, drug transport, and other cellular processes.

Large genomic databases provide an opportunity to investigate whether measurable genome-level characteristics can be used to distinguish resistant and susceptible bacterial isolates.

Machine learning provides a way to compare different computational approaches for this classification problem.

### 1.2 Research Question

> How do different machine learning algorithms compare in classifying ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes using genome-level features?

### 1.3 Objectives

1. Construct a dataset containing ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes.
2. Clean and prepare the genomic and phenotype data for machine learning.
3. Extract genome-level features from BV-BRC.
4. Train multiple machine learning classification algorithms.
5. Compare model performance using multiple evaluation metrics.
6. Examine which genomic features are associated with model predictions.
7. Compare model computational performance.
8. Identify limitations and possible directions for future research.

---

## 2. Dataset

### 2.1 Data Source

Phenotype and genome information were obtained from the **Bacterial and Viral Bioinformatics Resource Center (BV-BRC)**.

The study focused specifically on the antibiotic **ciprofloxacin**.

The phenotype groups were:

- Resistant
- Susceptible

### 2.2 Dataset Construction

The phenotype datasets were filtered to retain ciprofloxacin records classified as resistant or susceptible.

Genome IDs were used to connect phenotype information with genome-level information.

Genomes appearing in both the resistant and susceptible datasets were removed because they had conflicting labels for the purposes of this study.

Duplicate genome records were also removed so that each genome contributed a single classification example.

### 2.3 Final Dataset

After cleaning, the current dataset contained approximately:

| Category | Number of genomes |
|---|---:|
| Resistant | 9,540 |
| Susceptible | 9,926 |
| Total | 19,466 |

These values should be verified against the final program output before the study is finalized.

---

## 3. Genome-Level Features

The following genome-level features were used as machine learning variables:

- Genome length
- GC content
- Coding sequences (CDS)
- Ribosomal RNA (rRNA)
- Transfer RNA (tRNA)
- Number of contigs
- Contig N50
- Contig L50
- Hypothetical CDS
- Number of plasmids
- CheckM completeness
- CheckM contamination

The genome identifier was used for merging the datasets but was not used as a predictive feature.

The resistance phenotype was encoded as the target variable:

- `0` = Susceptible
- `1` = Resistant

---

## 4. Data Preprocessing

### 4.1 Conflicting Genomes

Genome IDs present in both the resistant and susceptible datasets were excluded to avoid assigning contradictory labels to the same genome.

### 4.2 Duplicate Removal

Duplicate genome records were removed based on Genome ID.

### 4.3 Missing Values

Missing numerical feature values were replaced using the median value of the corresponding feature.

### 4.4 Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

A fixed random state was used to make the split reproducible. Stratification was applied so that the resistant/susceptible class proportions were maintained between the training and testing sets.

---

## 5. Machine Learning Methods

Eight classification algorithms were evaluated.

### 5.1 Logistic Regression

Logistic Regression was used as a linear baseline classification model. Standardized features were used for this model.

### 5.2 Random Forest

Random Forest is an ensemble method that combines multiple decision trees to produce a classification result. The model was configured with 100 trees.

### 5.3 Decision Tree

A Decision Tree was used to evaluate a single tree-based classification approach.

### 5.4 Support Vector Machine

A Support Vector Machine (SVM) was used to classify the two phenotype groups. Standardized features were used.

### 5.5 K-Nearest Neighbors

K-Nearest Neighbors (KNN) classifies observations according to the labels of nearby observations in feature space. The model used five neighbors.

### 5.6 Gradient Boosting

Gradient Boosting was used as an ensemble method that builds sequential decision trees to improve predictive performance.

### 5.7 Gaussian Naive Bayes

Gaussian Naive Bayes was included as a probabilistic classification method.

### 5.8 XGBoost

XGBoost was included as a gradient-boosted tree-based algorithm. The model used 100 estimators.

---

## 6. Model Evaluation

### 6.1 Accuracy

Accuracy measures the proportion of test samples classified correctly.

### 6.2 Precision

Precision measures the proportion of predicted resistant samples that were actually resistant.

### 6.3 Recall

Recall measures the proportion of actual resistant samples that were correctly identified.

### 6.4 F1 Score

The F1 score combines precision and recall into a single metric.

### 6.5 ROC-AUC

ROC-AUC measures the model's ability to distinguish between resistant and susceptible classes across classification thresholds.

### 6.6 Confusion Matrix

Confusion matrices were generated to examine true positives, true negatives, false positives, and false negatives.

### 6.7 Cross-Validation

Five-fold cross-validation was used to assess model performance across multiple training subsets. Mean accuracy and standard deviation were recorded.

### 6.8 Computational Time

Training time and prediction time were measured for each algorithm to compare computational requirements.

---

## 7. Experimental Results

> **This section should be completed using the final output of the program. Do not enter estimated values.**

### 7.1 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD |
| Support Vector Machine | TBD | TBD | TBD | TBD | TBD |
| K-Nearest Neighbors | TBD | TBD | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD | TBD | TBD |
| Gaussian Naive Bayes | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

### 7.2 Cross-Validation

| Model | Mean Accuracy | Standard Deviation |
|---|---:|---:|
| Logistic Regression | TBD | TBD |
| Random Forest | TBD | TBD |
| Decision Tree | TBD | TBD |
| Support Vector Machine | TBD | TBD |
| K-Nearest Neighbors | TBD | TBD |
| Gradient Boosting | TBD | TBD |
| Gaussian Naive Bayes | TBD | TBD |
| XGBoost | TBD | TBD |

### 7.3 Computational Performance

| Model | Training Time | Prediction Time |
|---|---:|---:|
| Logistic Regression | TBD | TBD |
| Random Forest | TBD | TBD |
| Decision Tree | TBD | TBD |
| Support Vector Machine | TBD | TBD |
| K-Nearest Neighbors | TBD | TBD |
| Gradient Boosting | TBD | TBD |
| Gaussian Naive Bayes | TBD | TBD |
| XGBoost | TBD | TBD |

---

## 8. Feature Analysis

Feature analysis was performed using models that provide feature importance or coefficients.

The following approaches were included:

- Random Forest feature importance
- Decision Tree feature importance
- Gradient Boosting feature importance
- XGBoost feature importance
- Logistic Regression coefficients

These analyses are intended to identify which genome-level variables contribute most strongly to model predictions.

### 8.1 Feature Importance Results

> Add the actual feature-importance outputs and graphs after the final analysis.

| Feature | Observed importance / coefficient |
|---|---:|
| Genome length | TBD |
| GC content | TBD |
| CDS | TBD |
| rRNA | TBD |
| tRNA | TBD |
| Contigs | TBD |
| Contig N50 | TBD |
| Contig L50 | TBD |
| Hypothetical CDS | TBD |
| Plasmids | TBD |
| CheckM completeness | TBD |
| CheckM contamination | TBD |

---

## 9. Data Visualization

The study includes visualizations of:

1. Resistance distribution
2. Genome length distribution
3. GC content distribution
4. Feature correlation heatmap
5. Resistant versus susceptible feature distributions
6. Confusion matrices
7. Model comparison graphs

The model comparison graphs display:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

---

## 10. Discussion

The final discussion should be based on the observed experimental results.

Important questions to address include:

- Which models produced similar or different performance?
- How did ROC-AUC compare with accuracy?
- Did tree-based models behave differently from linear models?
- Which genomic features were consistently important?
- Were the models affected by differences between bacterial species?
- Did cross-validation support the test-set results?
- Were there meaningful differences in computational cost?
- What types of errors were most common?

The results should be interpreted as evidence from this dataset rather than as proof that the identified genome-level features directly cause ciprofloxacin resistance.

---

## 11. Limitations

### 11.1 Genome-Level Features

The models primarily use broad genome-level statistics rather than specific resistance genes, mutations, or gene variants. Therefore, a model may identify associations with genomic composition without identifying the biological mechanism responsible for ciprofloxacin resistance.

### 11.2 Species Composition

The dataset contains multiple bacterial species. Differences in genome structure and composition between species may influence the classification results. Species composition should therefore be considered when interpreting the findings.

### 11.3 Dataset Representation

The dataset is based on publicly available records and may not represent all bacterial populations or geographic regions.

### 11.4 External Validation

Performance on this dataset does not necessarily indicate how the models would perform on an independent external dataset.

### 11.5 Phenotype Data

Resistance classifications originate from the underlying public database and may reflect differences in testing conditions, standards, and measurement procedures.

---

## 12. Conclusion

This study establishes a machine learning framework for comparing different algorithms in the classification of ciprofloxacin-resistant and ciprofloxacin-susceptible bacterial genomes using genome-level features.

Eight machine learning algorithms were evaluated using multiple performance metrics, cross-validation, confusion matrices, feature analysis, and computational-time measurements.

The final conclusions will be added after the complete experimental results have been obtained and verified.

---

## 13. Future Work

Possible extensions of the study include:

1. Incorporating resistance-associated genes.
2. Incorporating known ciprofloxacin resistance mutations.
3. Adding k-mer or sequence-derived features.
4. Performing species-specific analyses.
5. Testing the models on independent datasets.
6. Comparing additional machine learning algorithms.
7. Applying hyperparameter optimization.
8. Investigating explainable machine learning methods.
9. Comparing genome-level features with gene-level predictors.
10. Evaluating whether models generalize across bacterial species and datasets.

---

## 14. Reproducibility

The analysis was implemented in Python.

Major libraries used include:

- pandas
- requests
- matplotlib
- seaborn
- scikit-learn
- XGBoost

The project contains scripts for dataset preparation, model training, evaluation, visualization, feature analysis, cross-validation, and model comparison.

The analysis uses a fixed random state for the main train-test split to improve reproducibility.

---

## 15. References

### Data Source

Bacterial and Viral Bioinformatics Resource Center (BV-BRC).

https://www.bv-brc.org/

### Machine Learning

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

### Additional References

Relevant literature on ciprofloxacin resistance, bacterial genomics, antimicrobial resistance, and machine learning-based AMR prediction should be added after the final literature review.
