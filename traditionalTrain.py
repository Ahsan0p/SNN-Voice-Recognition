import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib



# ==============================
# DATASET PATH (LOCAL)
# ==============================
DATASET_PATH = r"E:\CE-45\sem6\DSP\dataset"

# Optional: verify structure
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

print(f"Using LOCAL dataset path: {DATASET_PATH}")
selected_words = ["yes", "no"]


# ==============================
# LOAD AND PREPROCESS
# ==============================
def load_and_preprocess():
    signals = []
    labels = []

    MAX_SAMPLES_PER_CLASS = 500

    for word in selected_words:
        folder = os.path.join(DATASET_PATH, word)

        if not os.path.exists(folder):
            print(f"WARNING: Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        np.random.shuffle(files)
        files = files[:MAX_SAMPLES_PER_CLASS]

        print(f"{word}: using {len(files)} samples")

        for file in files:
            path = os.path.join(folder, file)
            try:
                signal, sr = librosa.load(path, sr=16000)

                # Fix length to exactly 1 second (16000 samples)
                if len(signal) < 16000:
                    signal = np.pad(signal, (0, 16000 - len(signal)))
                else:
                    signal = signal[:16000]

                signals.append(signal)
                labels.append(word)

            except Exception as e:
                print(f"  Error loading {file}: {e}")

    return signals, labels


def extract_mfcc(signal, sr=16000):
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=13,
        n_fft=512, hop_length=256
    )
    # Average across time frames -> shape (13,)
    return np.mean(mfcc.T, axis=0)


# ==============================
# MAIN
# ==============================
print("\nLoading data...")
signals, labels = load_and_preprocess()
print(f"Total samples loaded: {len(signals)}")

print("Extracting MFCC features...")
features = [extract_mfcc(s) for s in signals]
X = np.array(features)

# Encode labels: "no" -> 0, "yes" -> 1 (alphabetical by default)
le = LabelEncoder()
y = le.fit_transform(labels)
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# Train Logistic Regression
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTraditional ML Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model, encoder, and test data
joblib.dump(model, 'traditional_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')

print("\nSaved:")
print("  traditional_model.pkl  <- the trained LR model")
print("  label_encoder.pkl      <- yes/no <-> 0/1 mapping")
print("  test_data.pkl          <- 20% test set for evaluation")