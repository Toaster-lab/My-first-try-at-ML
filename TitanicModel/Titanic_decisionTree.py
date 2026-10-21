import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

N_TREES = 100
MAX_DEPTH = 7
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

THRESHOLD = 0.5
FIND_BEST_THRESHOLD = True

df = pd.read_csv(
    "TitanicModel/Titanic-Dataset.csv",
    usecols=["Age", "Sex", "Pclass", "Survived"]
)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = df.dropna(subset=["Survived"])

X = df[["Age", "Sex", "Pclass"]]
y = df["Survived"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

model = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

if FIND_BEST_THRESHOLD:
    val_probabilities = model.predict_proba(X_val)[:, 1]

    best_accuracy = -1

    for threshold in np.arange(0.1, 0.91, 0.01):
        predictions = (val_probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            THRESHOLD = threshold

    print(f"Best threshold: {THRESHOLD:.2f}")
    print(f"Validation accuracy: {best_accuracy * 100:.2f}%")

train_probabilities = model.predict_proba(X_train)[:, 1]
test_probabilities = model.predict_proba(X_test)[:, 1]

train_predictions = (train_probabilities >= THRESHOLD).astype(int)
test_predictions = (test_probabilities >= THRESHOLD).astype(int)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nThreshold used: {THRESHOLD:.2f}")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, test_predictions, zero_division=0):.3f}")

new_passenger = pd.DataFrame(
    [[25, 1, 1]],
    columns=["Age", "Sex", "Pclass"]
)

new_passenger = imputer.transform(new_passenger)

probability = model.predict_proba(new_passenger)[0, 1]
prediction = int(probability >= THRESHOLD)

print(f"\nSurvival probability: {probability * 100:.2f}%")
print(f"Predicted class: {prediction}")