import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import psutil
import tracemalloc


selected_words = ["yes", "no"]
MAX_SAMPLES_PER_CLASS = 500
TIME_STEPS = 30
HIDDEN_SIZE = 64
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        beta = 5.0
        grad_input = grad_output.clone()
        surrogate = 1 / (1 + beta * torch.abs(input - threshold)) ** 2
        return grad_input * surrogate, None


class SpikingNeuron(nn.Module):
    def __init__(self, size, threshold=0.5, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.mem = None
        self.spike_fn = SurrogateSpikeFunction.apply
        self.size = size

    def reset(self, batch_size=1, device='cpu'):
        self.mem = torch.zeros(batch_size, self.size, device=device)

    def forward(self, x):
        if self.mem is None:
            self.reset(batch_size=x.shape[0], device=x.device)
        self.mem = self.decay * self.mem.detach() + x
        spike = self.spike_fn(self.mem, self.threshold)
        self.mem = self.mem - spike * self.threshold
        return spike


class SNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.neuron = SpikingNeuron(out_features)

    def reset(self, batch_size=1, device='cpu'):
        self.neuron.reset(batch_size=batch_size, device=device)

    def forward(self, x, T=TIME_STEPS):
        self.reset(batch_size=x.shape[0], device=x.device)
        spike_sum = torch.zeros(x.shape[0], self.weights.shape[1], device=x.device)
        for t in range(T):
            cur = torch.matmul(x, self.weights) + self.bias
            spk = self.neuron(cur)
            spike_sum += spk
        return spike_sum / T


class CustomSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = SNNLayer(input_size, hidden_size)
        self.output = SNNLayer(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

    def reset(self, batch_size=1, device='cpu'):
        self.hidden.reset(batch_size=batch_size, device=device)
        self.output.reset(batch_size=batch_size, device=device)


DATASET_PATH = r"E:\CE-45\sem6\DSP\dataset"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

print(f"Using LOCAL dataset path: {DATASET_PATH}")


def load_data():
    X, y = [], []
    print(f"Loading balanced dataset ({MAX_SAMPLES_PER_CLASS} per class)...")

    for word in selected_words:
        folder = os.path.join(DATASET_PATH, word)

        if not os.path.exists(folder):
            print(f"WARNING: Missing folder: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        np.random.shuffle(files)
        files = files[:MAX_SAMPLES_PER_CLASS]
        print(f"  {word}: using {len(files)} samples")

        for file in files:
            path = os.path.join(folder, file)
            try:
                signal, sr = librosa.load(path, sr=16000)

                if len(signal) < 16000:
                    signal = np.pad(signal, (0, 16000 - len(signal)))
                else:
                    signal = signal[:16000]

                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                feature = np.mean(mfcc.T, axis=0)

                X.append(feature)
                y.append(word)

            except Exception as e:
                print(f"  Error loading {file}: {e}")

    return np.array(X), np.array(y)


def normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max


def train(model, X_train, y_train, X_test, y_test):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        perm = torch.randperm(X_train_t.size(0))
        total_loss = 0

        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(dim=1)
            acc = (preds == y_test_t).float().mean().item()

        if acc > best_acc:
            best_acc = acc

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    print(f"\nBest accuracy during training: {best_acc:.4f}")

    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(dim=1).cpu().numpy()

    return final_preds


def main():
    process = psutil.Process()
    tracemalloc.start()
    start_total_time = time.time()

    X, y_labels = load_data()
    print(f"\nTotal samples: {len(X)}")

    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    label_map = {word: i for i, word in enumerate(sorted(np.unique(y_labels)))}
    print(f"Label mapping: {label_map}")
    y = np.array([label_map[word] for word in y_labels])

    X_norm, X_min, X_max = normalize(X)
    print(f"Feature range after normalization: {X_norm.min():.3f} to {X_norm.max():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = CustomSNN(
        input_size=13,
        hidden_size=HIDDEN_SIZE,
        output_size=2
    )
    print(f"\nModel architecture: 13 -> {HIDDEN_SIZE} -> 2")

    torch.save(model.state_dict(), 'temp_model.pth')
    model_size_bytes = os.path.getsize('temp_model.pth')
    model_size_mb = model_size_bytes / (1024 * 1024)
    os.remove('temp_model.pth')

    cpu_percent_before = process.cpu_percent(interval=0.5)
    ram_mb_before = process.memory_info().rss / (1024 * 1024)

    print("\nStarting training...")
    start_train_time = time.time()
    y_pred = train(model, X_train, y_train, X_test, y_test)
    train_time = time.time() - start_train_time

    cpu_percent_after = process.cpu_percent(interval=0.5)
    ram_mb_after = process.memory_info().rss / (1024 * 1024)

    cpu_percent = (cpu_percent_before + cpu_percent_after) / 2
    ram_mb = max(ram_mb_before, ram_mb_after)

    X_test_20 = X_test[:20] if len(X_test) >= 20 else X_test
    y_test_20 = y_test[:20]

    latencies = []
    spike_rates = []
    silent_neurons_count = 0

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(X_test_20):
            sample_tensor = torch.tensor(sample.reshape(1, -1), dtype=torch.float32).to(DEVICE)

            start_time = time.time()

            model.reset(batch_size=1, device=DEVICE)
            x = sample_tensor
            x = model.hidden(x)

            hidden_spikes = model.hidden.neuron.mem if model.hidden.neuron.mem is not None else torch.zeros(1,
                                                                                                            HIDDEN_SIZE)
            spike_rate = (hidden_spikes > 0.5).float().mean().item()
            spike_rates.append(spike_rate)

            x = model.output(x)

            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

    avg_latency = np.mean(latencies)

    avg_spike_rate = np.mean(spike_rates)

    with torch.no_grad():
        model.reset(batch_size=len(X_test_20), device=DEVICE)
        X_test_20_tensor = torch.tensor(X_test_20, dtype=torch.float32).to(DEVICE)
        _ = model.hidden(X_test_20_tensor)
        all_spikes = model.hidden.neuron.mem
        silent_neurons = (all_spikes < 0.01).float().mean().item() * 100

    silent_percent = silent_neurons

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    y_pred_20 = []
    model.eval()
    with torch.no_grad():
        for sample in X_test_20:
            sample_tensor = torch.tensor(sample.reshape(1, -1), dtype=torch.float32).to(DEVICE)
            output = model(sample_tensor)
            pred = output.argmax(dim=1).item()
            y_pred_20.append(pred)

    correct_predictions_20 = np.sum(np.array(y_pred_20) == y_test_20[:len(y_pred_20)])

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = time.time() - start_total_time

    print("\n" + "=" * 60)
    print("SNN MODEL METRICS REPORT")
    print("=" * 60)

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
    print(f"  spike rate: {avg_spike_rate:.4f} (average firing rate)")
    print(f"  silent %: {silent_percent:.2f}% (neurons with spike rate < 0.01)")

    print("\n✔ Generalization:")
    print(f"  correct predictions out of 20: {correct_predictions_20}/{len(X_test_20)}")

    print("=" * 60)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'X_min': X_min,
        'X_max': X_max,
        'label_mapping': label_map,
        'input_size': 13,
        'hidden_size': HIDDEN_SIZE,
        'output_size': 2,
    }

    torch.save(checkpoint, 'snn_model_fast.pth')

    print("\nSaved: snn_model_fast.pth")
    print("  Contains: weights + X_min/X_max + label mapping + architecture info")
    print("  This file is now compatible with Test&Compere.py")


if __name__ == "__main__":
    main()