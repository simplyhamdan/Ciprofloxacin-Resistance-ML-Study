import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBClassifier


# Internal messages

internal_messages = []


def log_message(*messages):
    """Store internal messages without displaying them."""
    internal_messages.append(" ".join(str(message) for message in messages))


# 0. Ask whether genome data already exists locally

while True:
    existing_data = input(
        "Is the genome data already downloaded? (Y/N): "
    ).strip().upper()

    if existing_data == "Y":
        break

    elif existing_data == "N":
        break

    else:
        print("Invalid input. Please enter Y or N.")


print("\nPreparing dataset and training models...")


# 1. Load phenotype data

resistant_data = pd.read_csv(
    DATA_DIR / "ciprofloxacin_resistant.csv",
    dtype={"Genome ID": str}
)

susceptible_data = pd.read_csv(
    DATA_DIR / "ciprofloxacin_susceptible.csv",
    dtype={"Genome ID": str}
)


# 2. Remove conflicting genomes

resistant_genomes = set(resistant_data["Genome ID"])
susceptible_genomes = set(susceptible_data["Genome ID"])

overlap = resistant_genomes & susceptible_genomes

resistant_data = resistant_data[
    ~resistant_data["Genome ID"].isin(overlap)
]

susceptible_data = susceptible_data[
    ~susceptible_data["Genome ID"].isin(overlap)
]


# 3. Remove duplicate genomes

resistant_data = resistant_data.drop_duplicates(
    subset="Genome ID"
)

susceptible_data = susceptible_data.drop_duplicates(
    subset="Genome ID"
)


# 4. Check final dataset

log_message(
    "Final resistant genomes:",
    len(resistant_data)
)

log_message(
    "Final susceptible genomes:",
    len(susceptible_data)
)

log_message(
    "Total unique genomes:",
    len(resistant_data) + len(susceptible_data)
)


# 5. Get unique Genome IDs

genome_ids = sorted(
    set(resistant_data["Genome ID"]) |
    set(susceptible_data["Genome ID"])
)


# 6. Retrieve genome data

if existing_data == "Y":

    feature_data = pd.read_csv(
        DATA_DIR / "feature_data.csv",
        dtype={"genome_id": str}
    )

    log_message("\nUsing existing local genome data.")

else:

    genome_features = []

    for i in range(0, len(genome_ids), 100):

        batch = genome_ids[i:i + 100]

        conditions = ",".join(
            f"eq(genome_id,{genome_id})"
            for genome_id in batch
        )

        url = (
            f"https://www.bv-brc.org/api/genome/"
            f"?or({conditions})&limit(100)"
        )

        response = requests.get(url)

        if response.status_code == 200:
            genome_features.extend(response.json())
        else:
            log_message(
                "Request failed:",
                response.status_code
            )

        log_message(
            f"Retrieved {len(genome_features)} genomes"
        )


    log_message(
        "Total genomes retrieved:",
        len(genome_features)
    )


    # 7. Convert API data to DataFrame

    genome_data = pd.DataFrame(genome_features)

    genome_data.to_csv(
        DATA_DIR / "genome_features.csv",
        index=False
    )

    log_message(
        "Genome data shape:",
        genome_data.shape
    )

    log_message("\nAvailable columns:")

    log_message(
        genome_data.columns.tolist()
    )


    # 8. Select genomic features

    selected_features = [
        "genome_id",
        "genome_length",
        "gc_content",
        "cds",
        "rrna",
        "trna",
        "contigs",
        "contig_n50",
        "contig_l50",
        "hypothetical_cds",
        "plasmids",
        "checkm_completeness",
        "checkm_contamination"
    ]

    feature_data = genome_data[selected_features]

    feature_data.to_csv(
        DATA_DIR / "feature_data.csv",
        index=False
    )

    log_message("\nFeature data:")
    log_message(
        feature_data.head()
    )

    log_message("\nFeature data shape:")
    log_message(
        feature_data.shape
    )


# 9. Create resistance labels

resistant_labels = resistant_data[["Genome ID"]].copy()
resistant_labels["label"] = 1

susceptible_labels = susceptible_data[["Genome ID"]].copy()
susceptible_labels["label"] = 0


# 10. Combine resistance labels

labels = pd.concat(
    [resistant_labels, susceptible_labels],
    ignore_index=True
)

labels = labels.rename(
    columns={"Genome ID": "genome_id"}
)

log_message("\nLabel data:")
log_message(
    labels.head()
)

log_message("\nLabel counts:")
log_message(
    labels["label"].value_counts()
)


# 11. Merge genome features with resistance labels

dataset = feature_data.merge(
    labels,
    on="genome_id",
    how="inner"
)

log_message("\nFinal dataset:")
log_message(
    dataset.head()
)

log_message("\nFinal dataset shape:")
log_message(
    dataset.shape
)


# 12. Check for missing values

log_message("\nMissing values:")
log_message(
    dataset.isna().sum()
)


# 13. Check feature data types

log_message("\nData types:")
log_message(
    dataset.dtypes
)


# 14. Prepare features and target

X = dataset.drop(
    columns=["genome_id", "label"]
)

y = dataset["label"]

log_message("\nFeatures:")
log_message(
    X.columns.tolist()
)

log_message("\nTarget:")
log_message(
    y.name
)


# 15. Fill missing feature values with the median

X = X.fillna(X.median())

log_message("\nMissing values after filling:")
log_message(
    X.isna().sum()
)


# 16. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

log_message(
    "\nTraining samples:",
    len(X_train)
)

log_message(
    "Testing samples:",
    len(X_test)
)


# 17. Train a Logistic Regression model

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

log_message("\nModel trained successfully!")


# 18. Make predictions on the test set

y_pred = model.predict(X_test_scaled)

log_message("\nPredictions made successfully!")

log_message(
    "Number of predictions:",
    len(y_pred)
)


# 19. Evaluate Logistic Regression

accuracy = accuracy_score(y_test, y_pred)

logistic_report = classification_report(
    y_test,
    y_pred
)

logistic_cm = confusion_matrix(
    y_test,
    y_pred
)


# 20. Train a Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# 21. Make Random Forest predictions

rf_predictions = rf_model.predict(X_test)


# 22. Evaluate Random Forest

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_report = classification_report(
    y_test,
    rf_predictions
)

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

# 27. Decision Tree Model

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_predictions
)

dt_report = classification_report(
    y_test,
    dt_predictions
)

dt_cm = confusion_matrix(
    y_test,
    dt_predictions
)

# 28. Support Vector Machine Model

svm_model = SVC(
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

svm_predictions = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_report = classification_report(
    y_test,
    svm_predictions
)

svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)

# 29. K-Nearest Neighbors Model

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_predictions = knn_model.predict(
    X_test_scaled
)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

knn_report = classification_report(
    y_test,
    knn_predictions
)

knn_cm = confusion_matrix(
    y_test,
    knn_predictions
)

# 30. Gradient Boosting Model

gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_accuracy = accuracy_score(
    y_test,
    gb_predictions
)

gb_report = classification_report(
    y_test,
    gb_predictions
)

gb_cm = confusion_matrix(
    y_test,
    gb_predictions
)

# 31. Gaussian Naive Bayes Model

nb_model = GaussianNB()

nb_model.fit(
    X_train,
    y_train
)

nb_predictions = nb_model.predict(
    X_test
)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

nb_report = classification_report(
    y_test,
    nb_predictions
)

nb_cm = confusion_matrix(
    y_test,
    nb_predictions
)

# 32. XGBoost Model

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

xgb_report = classification_report(
    y_test,
    xgb_predictions
)

xgb_cm = confusion_matrix(
    y_test,
    xgb_predictions
)

# 33. ROC-AUC Evaluation

logistic_auc = roc_auc_score(
    y_test,
    model.predict_proba(X_test_scaled)[:, 1]
)

rf_auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test)[:, 1]
)

dt_auc = roc_auc_score(
    y_test,
    dt_model.predict_proba(X_test)[:, 1]
)

svm_auc = roc_auc_score(
    y_test,
    svm_model.decision_function(X_test_scaled)
)

knn_auc = roc_auc_score(
    y_test,
    knn_model.predict_proba(X_test_scaled)[:, 1]
)

gb_auc = roc_auc_score(
    y_test,
    gb_model.predict_proba(X_test)[:, 1]
)

nb_auc = roc_auc_score(
    y_test,
    nb_model.predict_proba(X_test)[:, 1]
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_model.predict_proba(X_test)[:, 1]
)


def cross_validation_analysis():

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Support Vector Machine": SVC(
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss"
        )
    }

    print("\n========================================")
    print("       5-FOLD CROSS-VALIDATION")
    print("========================================")

    for name, cv_model in models.items():

        if name in [
            "Logistic Regression",
            "Support Vector Machine",
            "K-Nearest Neighbors"
        ]:
            scores = cross_val_score(
                cv_model,
                X_train_scaled,
                y_train,
                cv=5,
                scoring="accuracy"
            )

        else:
            scores = cross_val_score(
                cv_model,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy"
            )

        print(f"\n{name}")
        print(f"Fold Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}")


def model_time_comparison():

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            X_train_scaled,
            X_test_scaled
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            X_train,
            X_test
        ),

        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=42
            ),
            X_train,
            X_test
        ),

        "Support Vector Machine": (
            SVC(
                random_state=42
            ),
            X_train_scaled,
            X_test_scaled
        ),

        "K-Nearest Neighbors": (
            KNeighborsClassifier(
                n_neighbors=5
            ),
            X_train_scaled,
            X_test_scaled
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(
                random_state=42
            ),
            X_train,
            X_test
        ),

        "Gaussian Naive Bayes": (
            GaussianNB(),
            X_train,
            X_test
        ),

        "XGBoost": (
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss"
            ),
            X_train,
            X_test
        )
    }

    results = []

    print("\n========================================")
    print("       MODEL TIME COMPARISON")
    print("========================================")

    for name, (timed_model, train_data, test_data) in models.items():

        start_time = time.perf_counter()

        timed_model.fit(
            train_data,
            y_train
        )

        training_time = time.perf_counter() - start_time

        start_time = time.perf_counter()

        timed_model.predict(
            test_data
        )

        prediction_time = time.perf_counter() - start_time

        results.append({
            "Model": name,
            "Training Time (s)": training_time,
            "Prediction Time (s)": prediction_time
        })

    time_results = pd.DataFrame(results)

    print(
        time_results.to_string(
            index=False,
            formatters={
                "Training Time (s)": "{:.4f}".format,
                "Prediction Time (s)": "{:.4f}".format
            }
        )
    )

    return time_results


# 23. Visualization functions

def resistance_distribution():
    """Plot the number of resistant and susceptible genomes."""

    counts = dataset["label"].value_counts().sort_index()

    labels = ["Susceptible", "Resistant"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)

    plt.title("Resistance Distribution")
    plt.xlabel("Resistance Status")
    plt.ylabel("Number of Genomes")

    plt.tight_layout()
    plt.show()


def genome_length_distribution():
    """Plot genome length distribution by resistance status."""

    plt.figure(figsize=(8, 5))

    plt.hist(
        dataset.loc[dataset["label"] == 0, "genome_length"],
        bins=50,
        alpha=0.6,
        label="Susceptible"
    )

    plt.hist(
        dataset.loc[dataset["label"] == 1, "genome_length"],
        bins=50,
        alpha=0.6,
        label="Resistant"
    )

    plt.title("Genome Length Distribution")
    plt.xlabel("Genome Length (bp)")
    plt.ylabel("Number of Genomes")
    plt.legend()

    plt.tight_layout()
    plt.show()


def gc_content_distribution():
    """Plot GC content distribution by resistance status."""

    plt.figure(figsize=(8, 5))

    plt.hist(
        dataset.loc[dataset["label"] == 0, "gc_content"],
        bins=50,
        alpha=0.6,
        label="Susceptible"
    )

    plt.hist(
        dataset.loc[dataset["label"] == 1, "gc_content"],
        bins=50,
        alpha=0.6,
        label="Resistant"
    )

    plt.title("GC Content Distribution")
    plt.xlabel("GC Content (%)")
    plt.ylabel("Number of Genomes")
    plt.legend()

    plt.tight_layout()
    plt.show()


def correlation_heatmap():
    """Plot correlation between numerical features."""

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        X.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def feature_comparison():
    """Compare feature distributions between resistant and susceptible genomes."""

    features = [
        "genome_length",
        "gc_content",
        "cds",
        "rrna",
        "trna",
        "contigs",
        "contig_n50",
        "contig_l50",
        "hypothetical_cds",
        "plasmids",
        "checkm_completeness",
        "checkm_contamination"
    ]

    for feature in features:

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [
                dataset.loc[
                    dataset["label"] == 0,
                    feature
                ],
                dataset.loc[
                    dataset["label"] == 1,
                    feature
                ]
            ],
            tick_labels=["Susceptible", "Resistant"]
        )

        plt.title(f"{feature} Comparison")
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(cm, title):
    """Display a confusion matrix."""

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Susceptible", "Resistant"],
        yticklabels=["Susceptible", "Resistant"]
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


def confusion_matrix_menu():

    while True:

        print("\n========================================")
        print("        CONFUSION MATRICES")
        print("========================================")
        print("[1] Logistic Regression")
        print("[2] Random Forest")
        print("[3] Decision Tree")
        print("[4] Support Vector Machine")
        print("[5] K-Nearest Neighbors")
        print("[6] Gradient Boosting")
        print("[7] Gaussian Naive Bayes")
        print("[8] XGBoost")
        print("[9] Back")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            plot_confusion_matrix(
                logistic_cm,
                "Logistic Regression Confusion Matrix"
            )

        elif choice == "2":
            plot_confusion_matrix(
                rf_cm,
                "Random Forest Confusion Matrix"
            )

        elif choice == "3":
            plot_confusion_matrix(
                dt_cm,
                "Decision Tree Confusion Matrix"
            )

        elif choice == "4":
            plot_confusion_matrix(
                svm_cm,
                "Support Vector Machine Confusion Matrix"
            )

        elif choice == "5":
            plot_confusion_matrix(
                knn_cm,
                "K-Nearest Neighbors Confusion Matrix"
            )

        elif choice == "6":
            plot_confusion_matrix(
                gb_cm,
                "Gradient Boosting Confusion Matrix"
            )

        elif choice == "7":
            plot_confusion_matrix(
                nb_cm,
                "Gaussian Naive Bayes Confusion Matrix"
            )

        elif choice == "8":
            plot_confusion_matrix(
                xgb_cm,
                "XGBoost Confusion Matrix"
            )

        elif choice == "9":
            break

        else:
            print("\nInvalid choice. Please enter 1-9.")


def feature_analysis_menu():

    while True:

        print("\n========================================")
        print("          FEATURE ANALYSIS")
        print("========================================")
        print("[1] Random Forest Importance")
        print("[2] Decision Tree Importance")
        print("[3] Gradient Boosting Importance")
        print("[4] XGBoost Importance")
        print("[5] Logistic Regression Coefficients")
        print("[6] Back")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":

            print("\n========================================")
            print("       RANDOM FOREST IMPORTANCE")
            print("========================================")

            importance = pd.Series(
                rf_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            plt.figure(figsize=(10, 6))

            importance.sort_values().plot(
                kind="barh"
            )

            plt.title("Random Forest Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.show()

        elif choice == "2":

            print("\n========================================")
            print("        DECISION TREE IMPORTANCE")
            print("========================================")

            importance = pd.Series(
                dt_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            plt.figure(figsize=(10, 6))

            importance.sort_values().plot(
                kind="barh"
            )

            plt.title("Decision Tree Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.show()

        elif choice == "3":

            print("\n========================================")
            print("      GRADIENT BOOSTING IMPORTANCE")
            print("========================================")

            importance = pd.Series(
                gb_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            plt.figure(figsize=(10, 6))

            importance.sort_values().plot(
                kind="barh"
            )

            plt.title("Gradient Boosting Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.show()

        elif choice == "4":

            print("\n========================================")
            print("          XGBOOST IMPORTANCE")
            print("========================================")

            importance = pd.Series(
                xgb_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            plt.figure(figsize=(10, 6))

            importance.sort_values().plot(
                kind="barh"
            )

            plt.title("XGBoost Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.show()

        elif choice == "5":

            print("\n========================================")
            print("     LOGISTIC REGRESSION COEFFICIENTS")
            print("========================================")

            coefficients = pd.Series(
                model.coef_[0],
                index=X.columns
            ).sort_values()

            print("\nFeature Coefficients:")
            print(coefficients)

            plt.figure(figsize=(10, 6))

            coefficients.plot(
                kind="barh"
            )

            plt.title("Logistic Regression Feature Coefficients")
            plt.xlabel("Coefficient")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.show()

        elif choice == "6":

            break

        else:

            print("\nInvalid choice. Please enter 1-6.")


# 24. Visualization Menu

def visualization_menu():

    while True:

        print("\n========================================")
        print("        VISUALIZATION MENU")
        print("========================================")
        print("[1] Resistance Distribution")
        print("[2] Genome Length Distribution")
        print("[3] GC Content Distribution")
        print("[4] Feature Correlation Heatmap")
        print("[5] Resistant vs Susceptible Features")
        print("[6] Confusion Matrices")
        print("[7] Model Comparison Graphs")
        print("[8] Back")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":

            resistance_distribution()

        elif choice == "2":

            genome_length_distribution()

        elif choice == "3":

            gc_content_distribution()

        elif choice == "4":

            correlation_heatmap()

        elif choice == "5":

            feature_comparison()

        elif choice == "6":

            confusion_matrix_menu()

        elif choice == "7":

            model_comparison_graphs()

        elif choice == "8":

            break

        else:

            print("\nInvalid choice. Please enter 1-8.")


def compare_models():

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "Gradient Boosting",
            "Gaussian Naive Bayes",
            "XGBoost"
        ],

        "Accuracy": [
            accuracy_score(y_test, y_pred),
            accuracy_score(y_test, rf_predictions),
            accuracy_score(y_test, dt_predictions),
            accuracy_score(y_test, svm_predictions),
            accuracy_score(y_test, knn_predictions),
            accuracy_score(y_test, gb_predictions),
            accuracy_score(y_test, nb_predictions),
            accuracy_score(y_test, xgb_predictions)
        ],

        "Precision": [
            classification_report(
                y_test, y_pred, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, rf_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, dt_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, svm_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, knn_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, gb_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, nb_predictions, output_dict=True
            )["weighted avg"]["precision"],

            classification_report(
                y_test, xgb_predictions, output_dict=True
            )["weighted avg"]["precision"]
        ],

        "Recall": [
            classification_report(
                y_test, y_pred, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, rf_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, dt_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, svm_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, knn_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, gb_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, nb_predictions, output_dict=True
            )["weighted avg"]["recall"],

            classification_report(
                y_test, xgb_predictions, output_dict=True
            )["weighted avg"]["recall"]
        ],

        "F1 Score": [
            classification_report(
                y_test, y_pred, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, rf_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, dt_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, svm_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, knn_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, gb_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, nb_predictions, output_dict=True
            )["weighted avg"]["f1-score"],

            classification_report(
                y_test, xgb_predictions, output_dict=True
            )["weighted avg"]["f1-score"]
        ],

        "ROC-AUC": [
            logistic_auc,
            rf_auc,
            dt_auc,
            svm_auc,
            knn_auc,
            gb_auc,
            nb_auc,
            xgb_auc
        ]
    })

    print("\n========================================")
    print("         MODEL COMPARISON")
    print("========================================")

    print(
        comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2%}".format,
                "Precision": "{:.2%}".format,
                "Recall": "{:.2%}".format,
                "F1 Score": "{:.2%}".format,
                "ROC-AUC": "{:.4f}".format
            }
        )
    )


def model_comparison_graphs():

    models = [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Gradient Boosting",
        "Gaussian Naive Bayes",
        "XGBoost"
    ]

    accuracy_scores = [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test, dt_predictions),
        accuracy_score(y_test, svm_predictions),
        accuracy_score(y_test, knn_predictions),
        accuracy_score(y_test, gb_predictions),
        accuracy_score(y_test, nb_predictions),
        accuracy_score(y_test, xgb_predictions)
    ]

    precision_scores = [
        classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, rf_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, dt_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, svm_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, knn_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, gb_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, nb_predictions, output_dict=True)["weighted avg"]["precision"],
        classification_report(y_test, xgb_predictions, output_dict=True)["weighted avg"]["precision"]
    ]

    recall_scores = [
        classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, rf_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, dt_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, svm_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, knn_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, gb_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, nb_predictions, output_dict=True)["weighted avg"]["recall"],
        classification_report(y_test, xgb_predictions, output_dict=True)["weighted avg"]["recall"]
    ]

    f1_scores = [
        classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, rf_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, dt_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, svm_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, knn_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, gb_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, nb_predictions, output_dict=True)["weighted avg"]["f1-score"],
        classification_report(y_test, xgb_predictions, output_dict=True)["weighted avg"]["f1-score"]
    ]

    auc_scores = [
        logistic_auc,
        rf_auc,
        dt_auc,
        svm_auc,
        knn_auc,
        gb_auc,
        nb_auc,
        xgb_auc
    ]

    metrics = {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1 Score": f1_scores,
        "ROC-AUC": auc_scores
    }

    for metric, scores in metrics.items():

        plt.figure(figsize=(10, 6))

        plt.bar(models, scores)

        plt.title(f"Model Comparison - {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()
        plt.show()


# 25. View Internal Messages

def view_internal_messages():

    print("\n========================================")
    print("        INTERNAL MESSAGES")
    print("========================================")

    for message in internal_messages:
        print(message)


# 28.(yes it is odd, but i made this after step 27) Learning Models Menu

def model_menu():

    while True:

        print("\n========================================")
        print("           LEARNING MODELS")
        print("========================================")
        print("[1] Logistic Regression")
        print("[2] Random Forest")
        print("[3] Decision Tree")
        print("[4] Support Vector Machine")
        print("[5] K-Nearest Neighbors")
        print("[6] Gradient Boosting")
        print("[7] Gaussian Naive Bayes")
        print("[8] XGBoost")
        print("[9] Compare Models")
        print("[10] Cross-Validation")
        print("[11] Model Time Comparison")
        print("[12] Back")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":

            print("\n========================================")
            print("       LOGISTIC REGRESSION")
            print("========================================")

            print(f"\nAccuracy: {accuracy:.2%}")
            print(f"ROC-AUC: {logistic_auc:.4f}")

            print("\nClassification Report:")
            print(logistic_report)

            print("\nConfusion Matrix:")
            print(logistic_cm)

        elif choice == "2":

            print("\n========================================")
            print("          RANDOM FOREST")
            print("========================================")

            print(f"\nAccuracy: {rf_accuracy:.2%}")
            print(f"ROC-AUC: {rf_auc:.4f}")

            print("\nClassification Report:")
            print(rf_report)

            print("\nConfusion Matrix:")
            print(rf_cm)

        elif choice == "3":

            print("\n========================================")
            print("          DECISION TREE")
            print("========================================")

            print(f"\nAccuracy: {dt_accuracy:.2%}")
            print(f"ROC-AUC: {dt_auc:.4f}")

            print("\nClassification Report:")
            print(dt_report)

            print("\nConfusion Matrix:")
            print(dt_cm)

        elif choice == "4":

            print("\n========================================")
            print("       SUPPORT VECTOR MACHINE")
            print("========================================")

            print(f"\nAccuracy: {svm_accuracy:.2%}")
            print(f"ROC-AUC: {svm_auc:.4f}")

            print("\nClassification Report:")
            print(svm_report)

            print("\nConfusion Matrix:")
            print(svm_cm)

        elif choice == "5":

            print("\n========================================")
            print("       K-NEAREST NEIGHBORS")
            print("========================================")

            print(f"\nAccuracy: {knn_accuracy:.2%}")
            print(f"ROC-AUC: {knn_auc:.4f}")

            print("\nClassification Report:")
            print(knn_report)

            print("\nConfusion Matrix:")
            print(knn_cm)

        elif choice == "6":

            print("\n========================================")
            print("        GRADIENT BOOSTING")
            print("========================================")

            print(f"\nAccuracy: {gb_accuracy:.2%}")
            print(f"ROC-AUC: {gb_auc:.4f}")

            print("\nClassification Report:")
            print(gb_report)

            print("\nConfusion Matrix:")
            print(gb_cm)

        elif choice == "7":

            print("\n========================================")
            print("       GAUSSIAN NAIVE BAYES")
            print("========================================")

            print(f"\nAccuracy: {nb_accuracy:.2%}")
            print(f"ROC-AUC: {nb_auc:.4f}")

            print("\nClassification Report:")
            print(nb_report)

            print("\nConfusion Matrix:")
            print(nb_cm)

        elif choice == "8":

            print("\n========================================")
            print("              XGBOOST")
            print("========================================")

            print(f"\nAccuracy: {xgb_accuracy:.2%}")
            print(f"ROC-AUC: {xgb_auc:.4f}")

            print("\nClassification Report:")
            print(xgb_report)

            print("\nConfusion Matrix:")
            print(xgb_cm)

        elif choice == "9":

            compare_models()

        elif choice == "10":

            cross_validation_analysis()

        elif choice == "11":
            model_time_comparison() 

        elif choice == "12":
            break            

        else:

            print("\nInvalid choice. Please enter 1-12.")


# 26. Main Menu

print("\nModels ready.")

while True:

    print("\n========================================")
    print("      CIPROFLOXACIN RESISTANCE ML STUDY")
    print("========================================")
    print("[1] Machine Learning Models")
    print("[2] Visualize Dataset")
    print("[3] Feature Analysis")
    print("[4] View Internal Messages")
    print("[5] Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":

        model_menu()

    elif choice == "2":

        visualization_menu()

    elif choice == "3":

        feature_analysis_menu()

    elif choice == "4":

        view_internal_messages()

    elif choice == "5":

        print("\nExiting program...")
        break

    else:

        print("\nInvalid choice. Please enter 1-5.")