import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import time
import psutil
import tracemalloc



DATASET_PATH = r"E:\CE-45\sem6\DSP\dataset"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

print(f"Using LOCAL dataset path: {DATASET_PATH}")
selected_words = ["yes", "no"]


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
    return np.mean(mfcc.T, axis=0)


print("\nLoading data...")
signals, labels = load_and_preprocess()
print(f"Total samples loaded: {len(signals)}")

print("Extracting MFCC features...")
features = [extract_mfcc(s) for s in signals]
X = np.array(features)

le = LabelEncoder()
y = le.fit_transform(labels)
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

process = psutil.Process()
tracemalloc.start()

test_indices = np.random.choice(len(X_test), min(20, len(X_test)), replace=False)
X_test_20 = X_test[test_indices]
y_test_20 = y_test[test_indices]

print("\nTraining Logistic Regression...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
training_time = time.time() - start_time

model_size_bytes = os.path.getsize('traditional_model.pkl') if os.path.exists('traditional_model.pkl') else 0
joblib.dump(model, 'traditional_model.pkl')
model_size_bytes = os.path.getsize('traditional_model.pkl')
model_size_mb = model_size_bytes / (1024 * 1024)

cpu_percent = process.cpu_percent(interval=1)
ram_mb = process.memory_info().rss / (1024 * 1024)

latencies = []
for sample in X_test_20:
    start_time = time.time()
    _ = model.predict(sample.reshape(1, -1))
    latencies.append(time.time() - start_time)
avg_latency = np.mean(latencies) * 1000

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

y_pred_20 = model.predict(X_test_20)
correct_predictions_20 = np.sum(y_pred_20 == y_test_20)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\n" + "="*60)
print("TRADITIONAL MODEL METRICS REPORT")
print("="*60)

print("\n✔ Metrics:")
print(f"  accuracy: {accuracy:.4f}")
print(f"  precision: {precision:.4f}")
print(f"  recall: {recall:.4f}")
print(f"  f1: {f1:.4f}")

print("\n✔ Performance:")
print(f"  latency: {avg_latency:.2f} ms (average over 20 samples)")
print(f"  CPU %: {cpu_percent:.1f}%")
print(f"  RAM usage: {ram_mb:.2f} MB")
print(f"  model size: {model_size_mb:.2f} MB")

print("\n✔ SNN-specific:")
print(f"  spike rate: N/A (not applicable for traditional model)")
print(f"  silent %: N/A (not applicable for traditional model)")

print("\n✔ Generalization:")
print(f"  correct predictions out of 20: {correct_predictions_20}/20")

print("="*60)

print(f"\nTraditional ML Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, 'traditional_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')

print("\nSaved:")
print("  traditional_model.pkl  <- the trained LR model")
print("  label_encoder.pkl      <- yes/no <-> 0/1 mapping")
print("  test_data.pkl          <- 20% test set for evaluation")