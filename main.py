import pandas as pd
import requests


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
    "Bacterial-Antibiotic-Resistance-Predictor/data/ciprofloxacin_resistant.csv",
    dtype={"Genome ID": str}
)

susceptible_data = pd.read_csv(
    "Bacterial-Antibiotic-Resistance-Predictor/data/ciprofloxacin_susceptible.csv",
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

genome_ids = list(
    set(resistant_data["Genome ID"]) |
    set(susceptible_data["Genome ID"])
)


# 6. Retrieve genome data

if existing_data == "Y":

    feature_data = pd.read_csv(
        "Bacterial-Antibiotic-Resistance-Predictor/data/feature_data.csv",
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
        "Bacterial-Antibiotic-Resistance-Predictor/data/genome_features.csv",
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
        "Bacterial-Antibiotic-Resistance-Predictor/data/feature_data.csv",
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

from sklearn.model_selection import train_test_split

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

from sklearn.ensemble import RandomForestClassifier

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


# 23. Visualization functions

import matplotlib.pyplot as plt
import seaborn as sns


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


def random_forest_importance():
    """Display Random Forest feature importance."""

    importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    plt.figure(figsize=(9, 6))

    importance.plot(
        kind="barh"
    )

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.show()


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
        print("[6] Logistic Regression Confusion Matrix")
        print("[7] Random Forest Confusion Matrix")
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
            plot_confusion_matrix(
                logistic_cm,
                "Logistic Regression Confusion Matrix"
            )

        elif choice == "7":
            plot_confusion_matrix(
                rf_cm,
                "Random Forest Confusion Matrix"
            )

        elif choice == "8":
            break

        else:
            print("\nInvalid choice. Please enter 1-8.")


# 25. View Internal Messages

def view_internal_messages():

    print("\n========================================")
    print("        INTERNAL MESSAGES")
    print("========================================")

    for message in internal_messages:
        print(message)


# 26. Main Menu

print("\nModels ready.")

while True:

    print("\n========================================")
    print("   ANTIBIOTIC RESISTANCE PREDICTOR")
    print("========================================")
    print("[1] Logistic Regression")
    print("[2] Random Forest")
    print("[3] Compare Both Models")
    print("[4] Visualize Dataset")
    print("[5] Feature Importance")
    print("[6] View Internal Messages")
    print("[7] Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":

        print("\n========================================")
        print("       LOGISTIC REGRESSION")
        print("========================================")

        print(f"\nAccuracy: {accuracy:.2%}")

        print("\nClassification Report:")
        print(logistic_report)

        print("\nConfusion Matrix:")
        print(logistic_cm)

    elif choice == "2":

        print("\n========================================")
        print("          RANDOM FOREST")
        print("========================================")

        print(f"\nAccuracy: {rf_accuracy:.2%}")

        print("\nClassification Report:")
        print(rf_report)

        print("\nConfusion Matrix:")
        print(rf_cm)

    elif choice == "3":

        print("\n========================================")
        print("         MODEL COMPARISON")
        print("========================================")

        print(
            f"\nLogistic Regression Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Random Forest Accuracy:       "
            f"{rf_accuracy:.2%}"
        )

        if rf_accuracy > accuracy:

            print("\nRandom Forest performed better.")

        elif accuracy > rf_accuracy:

            print("\nLogistic Regression performed better.")

        else:

            print("\nBoth models performed equally.")

    elif choice == "4":

        visualization_menu()

    elif choice == "5":

        print("\n========================================")
        print("       RANDOM FOREST FEATURE IMPORTANCE")
        print("========================================")

        random_forest_importance()

    elif choice == "6":

        view_internal_messages()

    elif choice == "7":

        print("\nExiting program...")
        break

    else:

        print("\nInvalid choice. Please enter 1-7.")