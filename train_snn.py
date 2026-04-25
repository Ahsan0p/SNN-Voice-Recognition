import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from sklearn.model_selection import train_test_split
import joblib


# ==============================
# CONFIG
# ==============================
selected_words = ["yes", "no"]
MAX_SAMPLES_PER_CLASS = 500
TIME_STEPS = 30       # Matches Test&Compere.py (T=30 in SNNLayer.forward)
HIDDEN_SIZE = 64
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ==============================
# SURROGATE SPIKE FUNCTION
# ==============================
# Problem: the real spike function (step function) has zero gradient everywhere.
# Backpropagation needs a gradient to learn. Solution: use a smooth fake
# gradient only during the backward pass. This is called a surrogate gradient.
class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        # Forward pass: real spikes (0 or 1)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        beta = 5.0
        grad_input = grad_output.clone()
        # Backward pass: smooth approximation (surrogate)
        surrogate = 1 / (1 + beta * torch.abs(input - threshold)) ** 2
        return grad_input * surrogate, None


# ==============================
# LIF (LEAKY INTEGRATE-AND-FIRE) NEURON
# ==============================
# Holds its own internal membrane potential state.
# Each call to forward() is one timestep.
class SpikingNeuron(nn.Module):
    def __init__(self, size, threshold=0.5, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.mem = None          # Membrane potential (internal state)
        self.spike_fn = SurrogateSpikeFunction.apply
        self.size = size

    def reset(self, batch_size=1, device='cpu'):
        """Reset membrane potential to zero. Call before each new batch."""
        self.mem = torch.zeros(batch_size, self.size, device=device)

    def forward(self, x):
        if self.mem is None:
            self.reset(batch_size=x.shape[0], device=x.device)
        # FIX: detach mem from the previous timestep's computation graph.
        # Without this, PyTorch chains all 30 timesteps into one graph.
        # When backward() frees that graph after the first batch, the next
        # batch tries to backward through already-freed tensors -> crash.
        # detach() cuts the chain: each timestep's graph is independent.
        self.mem = self.decay * self.mem.detach() + x
        # Fire: emit spike if membrane crossed threshold
        spike = self.spike_fn(self.mem, self.threshold)
        # Reset: subtract threshold worth of potential after firing
        self.mem = self.mem - spike * self.threshold
        return spike


# ==============================
# SNN LAYER
# ==============================
# One fully-connected layer of spiking neurons.
# Runs the same input through T timesteps and returns average spike rate.
class SNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.neuron = SpikingNeuron(out_features)

    def reset(self, batch_size=1, device='cpu'):
        self.neuron.reset(batch_size=batch_size, device=device)

    def forward(self, x, T=TIME_STEPS):
        self.reset(batch_size=x.shape[0], device=x.device)  # reset at start of each forward pass
        spike_sum = torch.zeros(x.shape[0], self.weights.shape[1], device=x.device)
        for t in range(T):
            cur = torch.matmul(x, self.weights) + self.bias
            spk = self.neuron(cur)
            spike_sum += spk
        # Return average firing rate over all timesteps
        return spike_sum / T


# ==============================
# FULL SNN MODEL (CustomSNN)
# ==============================
# This is the SAME architecture used in Test&Compere.py.
# Training and testing must use identical architecture or weights won't load.
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


# ==============================
# DATASET DOWNLOAD
# ==============================
# ==============================
# DATASET PATH (LOCAL)
# ==============================
DATASET_PATH = r"E:\CE-45\sem6\DSP\dataset"

# Optional: verify structure
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

print(f"Using LOCAL dataset path: {DATASET_PATH}")


# ==============================
# DATA LOADING
# ==============================
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

                # Pad or trim to exactly 1 second
                if len(signal) < 16000:
                    signal = np.pad(signal, (0, 16000 - len(signal)))
                else:
                    signal = signal[:16000]

                # Extract 13 MFCC features (averaged over time)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                feature = np.mean(mfcc.T, axis=0)

                X.append(feature)
                y.append(word)

            except Exception as e:
                print(f"  Error loading {file}: {e}")

    return np.array(X), np.array(y)


def normalize(X):
    """Scale features to [0, 1] range. Required for spike rate coding."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max


# ==============================
# TRAINING LOOP
# ==============================
def train(model, X_train, y_train, X_test, y_test):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(DEVICE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        # Shuffle training data each epoch
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

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(dim=1)
            acc = (preds == y_test_t).float().mean().item()

        if acc > best_acc:
            best_acc = acc

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    print(f"\nBest accuracy during training: {best_acc:.4f}")


# ==============================
# MAIN
# ==============================
def main():
    # Load data
    X, y_labels = load_data()
    print(f"\nTotal samples: {len(X)}")

    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Build label map: word -> integer index
    # sorted() ensures consistent ordering across runs
    label_map = {word: i for i, word in enumerate(sorted(np.unique(y_labels)))}
    print(f"Label mapping: {label_map}")
    y = np.array([label_map[word] for word in y_labels])

    # Normalize to [0, 1] and keep min/max for inference time
    X_norm, X_min, X_max = normalize(X)
    print(f"Feature range after normalization: {X_norm.min():.3f} to {X_norm.max():.3f}")

    # Train/test split (stratified = equal class balance in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model — SAME architecture as Test&Compere.py CustomSNN
    model = CustomSNN(
        input_size=13,
        hidden_size=HIDDEN_SIZE,
        output_size=2
    )
    print(f"\nModel architecture: 13 -> {HIDDEN_SIZE} -> 2")

    # Train
    print("\nStarting training...")
    train(model, X_train, y_train, X_test, y_test)

    # ==============================
    # SAVE FULL CHECKPOINT
    # ==============================
    # IMPORTANT: Save everything the test file needs in ONE dictionary.
    # Test&Compere.py expects: model_state_dict, X_min, X_max, label_mapping,
    # input_size, hidden_size, output_size
    checkpoint = {
        'model_state_dict': model.state_dict(),   # Trained weights
        'X_min': X_min,                            # For normalizing new samples
        'X_max': X_max,                            # For normalizing new samples
        'label_mapping': label_map,                # e.g. {'no': 0, 'yes': 1}
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