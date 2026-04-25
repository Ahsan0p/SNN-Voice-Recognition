import os
import numpy as np
import torch
import torch.nn as nn
import librosa  # FIXED: Changed from librosa1 to librosa
import sounddevice as sd
import soundfile as sf
import joblib
import time
from tabulate import tabulate
import matplotlib.pyplot as plt


# ==============================
# EXACT SNN ARCHITECTURE (MATCHING TRAINING)
# ==============================
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

    def reset(self):
        self.mem = torch.zeros(self.size)

    def forward(self, x):
        if self.mem is None:
            self.reset()
        self.mem = self.decay * self.mem + x
        spike = self.spike_fn(self.mem, self.threshold)
        self.mem = self.mem - spike * self.threshold
        return spike


class SNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.neuron = SpikingNeuron(out_features)

    def reset(self):
        self.neuron.reset()

    def forward(self, x, T=30):
        # FIXED: Use same device as input
        spike_sum = torch.zeros(x.shape[0], self.weights.shape[1], device=x.device)
        spike_counts = []  # Track spikes per timestep

        for t in range(T):
            cur = torch.matmul(x, self.weights) + self.bias
            spk = self.neuron(cur)
            spike_sum += spk
            spike_counts.append(spk.detach().cpu().numpy())

        return spike_sum / T, spike_counts


class CustomSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = SNNLayer(input_size, hidden_size)
        self.output = SNNLayer(hidden_size, output_size)
        self.timesteps = 30  # Store timesteps for spike tracking

    def forward(self, x):
        hidden_out, hidden_spikes = self.hidden(x)
        output_out, output_spikes = self.output(hidden_out)
        return output_out, hidden_spikes, output_spikes

    def reset(self):
        self.hidden.reset()
        self.output.reset()


# ==============================
# AUDIO PROCESSING
# ==============================
def record_audio(duration=1.0, sample_rate=16000):
    """Record from microphone"""
    print(f"\n🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
    sd.wait()
    recording = recording.flatten()
    print("✅ Recording complete!")
    return recording


def extract_features(audio, sr=16000):
    """Extract MFCC features (same as training)"""
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    else:
        audio = audio[:sr]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    features = np.mean(mfcc.T, axis=0)
    return features


# ==============================
# LOAD MODELS (FIXED VERSION)
# ==============================
def load_traditional_model():
    """Load Logistic Regression model"""
    try:
        model = joblib.load('traditional_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        print("✅ Traditional model loaded")
        return model, encoder
    except Exception as e:
        print(f"❌ Traditional model error: {e}")
        return None, None


def load_snn_model():
    """Load SNN model with ALL fixes applied"""
    try:
        # FIX 1: Correct filename
        checkpoint = torch.load(
            'snn_model_fast.pth',  # Changed from snn_model_fixed.pth
            map_location='cpu',
            weights_only=False  # FIX: Required for PyTorch 2.6+
        )

        # Extract metadata
        input_size = checkpoint.get('input_size', 13)
        hidden_size = checkpoint.get('hidden_size', 64)
        output_size = checkpoint.get('output_size', 2)

        # Create model with EXACT same architecture
        model = CustomSNN(input_size, hidden_size, output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get normalization parameters
        X_min = checkpoint.get('X_min', None)
        X_max = checkpoint.get('X_max', None)

        # FIX 2: Correct label mapping reversal
        label_mapping = checkpoint.get('label_mapping', {'yes': 0, 'no': 1})
        # Reverse mapping: index -> label
        idx_to_label = {v: k for k, v in label_mapping.items()}

        print(f"✅ SNN model loaded successfully")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Label mapping: {idx_to_label}")

        return model, X_min, X_max, idx_to_label

    except FileNotFoundError:
        print("❌ SNN model file 'snn_model_fast.pth' not found!")
        print("   Make sure you trained the model first and saved with this name")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error loading SNN model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ==============================
# PREDICTIONS WITH SNN METRICS
# ==============================
def predict_traditional(model, encoder, features):
    """Traditional model prediction"""
    start = time.time()
    features_2d = features.reshape(1, -1)
    pred = model.predict(features_2d)[0]
    probs = model.predict_proba(features_2d)[0]
    confidence = np.max(probs) * 100
    inference_time = (time.time() - start) * 1000

    word = encoder.inverse_transform([pred])[0]
    return word, confidence, inference_time


def calculate_spike_rate(hidden_spikes, output_spikes, timesteps=30):
    """Calculate average spike rate across all neurons and timesteps"""
    # Convert spike lists to numpy arrays
    hidden_spikes_array = np.array(hidden_spikes)  # Shape: (timesteps, batch, neurons)
    output_spikes_array = np.array(output_spikes)

    # Count total spikes
    total_spikes = np.sum(hidden_spikes_array) + np.sum(output_spikes_array)

    # Calculate total neurons
    hidden_neurons = hidden_spikes_array.shape[2] if len(hidden_spikes_array.shape) > 2 else 0
    output_neurons = output_spikes_array.shape[2] if len(output_spikes_array.shape) > 2 else 0
    total_neurons = hidden_neurons + output_neurons

    # Calculate spike rate (spikes per neuron per timestep)
    if total_neurons > 0:
        spike_rate = total_spikes / (total_neurons * timesteps)
    else:
        spike_rate = 0

    return spike_rate, total_spikes


def estimate_energy(spike_rate, total_spikes, features, num_classes=2):
    """Estimate energy consumption for SNN vs Traditional ANN"""
    # Energy constants (approximate research values)
    ENERGY_PER_SPIKE = 1.0  # Arbitrary unit for SNN
    ENERGY_PER_MAC = 10.0  # Traditional ANN requires much more energy

    # SNN energy (proportional to spikes)
    snn_energy = total_spikes * ENERGY_PER_SPIKE

    # Traditional model energy (based on MAC operations)
    num_features = len(features)
    trad_energy = num_features * num_classes * ENERGY_PER_MAC

    return snn_energy, trad_energy


def predict_snn(model, features, X_min, X_max, idx_to_label):
    """SNN prediction with spike rate and energy tracking"""
    start = time.time()

    # Normalize features (CRITICAL - same as training)
    if X_min is not None and X_max is not None:
        features = (features - X_min) / (X_max - X_min + 1e-8)

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Predict with spike tracking
    with torch.no_grad():
        model.reset()  # Reset neuron states
        output, hidden_spikes, output_spikes = model(x)
        probs = torch.softmax(output, dim=1)
        confidence = torch.max(probs).item() * 100
        pred_idx = torch.argmax(output, dim=1).item()

    inference_time = (time.time() - start) * 1000

    # Calculate spike metrics
    spike_rate, total_spikes = calculate_spike_rate(hidden_spikes, output_spikes, timesteps=model.timesteps)

    # Estimate energy
    snn_energy, trad_energy = estimate_energy(spike_rate, total_spikes, features)

    # FIXED: Use reversed mapping
    word = idx_to_label.get(pred_idx, 'unknown')
    return word, confidence, inference_time, spike_rate, total_spikes, snn_energy, trad_energy


# ==============================
# RESULTS TRACKING WITH ENHANCED METRICS
# ==============================
class ComparisonTracker:
    def __init__(self):
        self.results = []

    def add_result(self, true_label, trad_pred, snn_pred, trad_conf, snn_conf,
                   trad_time, snn_time, spike_rate, snn_energy, trad_energy):
        self.results.append({
            'true': true_label,
            'trad_pred': trad_pred,
            'snn_pred': snn_pred,
            'trad_correct': trad_pred == true_label,
            'snn_correct': snn_pred == true_label,
            'trad_conf': trad_conf,
            'snn_conf': snn_conf,
            'trad_time': trad_time,
            'snn_time': snn_time,
            'spike_rate': spike_rate,
            'snn_energy': snn_energy,
            'trad_energy': trad_energy
        })

    def get_metrics(self):
        if not self.results:
            return None

        n = len(self.results)
        trad_correct = sum(r['trad_correct'] for r in self.results)
        snn_correct = sum(r['snn_correct'] for r in self.results)

        return {
            'n': n,
            'trad': {
                'accuracy': trad_correct / n * 100,
                'avg_time': np.mean([r['trad_time'] for r in self.results]),
                'avg_conf': np.mean([r['trad_conf'] for r in self.results]),
                'correct': trad_correct,
                'avg_energy': np.mean([r['trad_energy'] for r in self.results])
            },
            'snn': {
                'accuracy': snn_correct / n * 100,
                'avg_time': np.mean([r['snn_time'] for r in self.results]),
                'avg_conf': np.mean([r['snn_conf'] for r in self.results]),
                'correct': snn_correct,
                'avg_spike_rate': np.mean([r['spike_rate'] for r in self.results]),
                'avg_energy': np.mean([r['snn_energy'] for r in self.results])
            }
        }

    def get_confusion_matrices(self):
        """Get confusion matrices for both models"""
        trad_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        snn_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for r in self.results:
            # Traditional
            if r['true'] == 'yes' and r['trad_pred'] == 'yes':
                trad_cm['TP'] += 1
            elif r['true'] == 'yes' and r['trad_pred'] == 'no':
                trad_cm['FN'] += 1
            elif r['true'] == 'no' and r['trad_pred'] == 'yes':
                trad_cm['FP'] += 1
            elif r['true'] == 'no' and r['trad_pred'] == 'no':
                trad_cm['TN'] += 1

            # SNN
            if r['true'] == 'yes' and r['snn_pred'] == 'yes':
                snn_cm['TP'] += 1
            elif r['true'] == 'yes' and r['snn_pred'] == 'no':
                snn_cm['FN'] += 1
            elif r['true'] == 'no' and r['snn_pred'] == 'yes':
                snn_cm['FP'] += 1
            elif r['true'] == 'no' and r['snn_pred'] == 'no':
                snn_cm['TN'] += 1

        return trad_cm, snn_cm


# ==============================
# DISPLAY FUNCTIONS
# ==============================
def print_results(true_label, trad_word, trad_conf, trad_time, trad_correct,
                  snn_word, snn_conf, snn_time, snn_correct,
                  spike_rate, snn_energy, trad_energy):
    """Display individual prediction results with SNN metrics"""
    print("\n" + "=" * 70)
    print("🎯 PREDICTION RESULTS")
    print("=" * 70)
    print(f"🎤 Actual: {true_label.upper()}")
    print("-" * 70)

    # Traditional model
    trad_symbol = "✅" if trad_correct else "❌"
    print(f"📊 TRADITIONAL (Logistic Regression):")
    print(f"   Prediction: {trad_word.upper()} {trad_symbol}")
    print(f"   Confidence: {trad_conf:.2f}%")
    print(f"   Time: {trad_time:.2f} ms")
    print(f"   Energy (est.): {trad_energy:.2f} units")

    print("-" * 70)

    # SNN model
    snn_symbol = "✅" if snn_correct else "❌"
    print(f"🧠 SNN (Spiking Neural Network):")
    print(f"   Prediction: {snn_word.upper()} {snn_symbol}")
    print(f"   Confidence: {snn_conf:.2f}%")
    print(f"   Time: {snn_time:.2f} ms")
    print(f"   Spike Rate: {spike_rate:.4f} (spikes/neuron/timestep)")
    print(f"   Energy (est.): {snn_energy:.2f} units")

    print("=" * 70)


def print_detailed_report(metrics, trad_cm, snn_cm):
    """Print comprehensive comparison report with SNN metrics"""
    print("\n" + "=" * 80)
    print("📊 DETAILED MODEL COMPARISON REPORT")
    print("=" * 80)

    # Main metrics table with SNN-specific metrics
    table = [
        ["Metric", "Traditional (LR)", "SNN", "Winner"],
        ["-" * 30, "-" * 20, "-" * 20, "-" * 15],
        ["Accuracy", f"{metrics['trad']['accuracy']:.2f}%",
         f"{metrics['snn']['accuracy']:.2f}%",
         "Traditional" if metrics['trad']['accuracy'] > metrics['snn']['accuracy']
         else "SNN" if metrics['snn']['accuracy'] > metrics['trad']['accuracy']
         else "Tie"],
        ["Avg Inference Time", f"{metrics['trad']['avg_time']:.2f} ms",
         f"{metrics['snn']['avg_time']:.2f} ms",
         "Traditional" if metrics['trad']['avg_time'] < metrics['snn']['avg_time']
         else "SNN" if metrics['snn']['avg_time'] < metrics['trad']['avg_time']
         else "Tie"],
        ["Avg Confidence", f"{metrics['trad']['avg_conf']:.2f}%",
         f"{metrics['snn']['avg_conf']:.2f}%",
         "Traditional" if metrics['trad']['avg_conf'] > metrics['snn']['avg_conf']
         else "SNN" if metrics['snn']['avg_conf'] > metrics['trad']['avg_conf']
         else "Tie"],
        ["Avg Spike Rate", "-", f"{metrics['snn']['avg_spike_rate']:.4f}", "Lower is better (SNN)"],
        ["Avg Energy (est.)", f"{metrics['trad']['avg_energy']:.2f}",
         f"{metrics['snn']['avg_energy']:.2f}",
         "SNN" if metrics['snn']['avg_energy'] < metrics['trad']['avg_energy']
         else "Traditional"],
        ["Correct/Total", f"{metrics['trad']['correct']}/{metrics['n']}",
         f"{metrics['snn']['correct']}/{metrics['n']}", "-"]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    # Confusion matrices
    print("\n📈 CONFUSION MATRICES:")
    print("-" * 50)

    print("\nTraditional Model:")
    print(f"   True Positives (yes→yes):  {trad_cm['TP']}")
    print(f"   True Negatives (no→no):    {trad_cm['TN']}")
    print(f"   False Positives (no→yes):  {trad_cm['FP']}")
    print(f"   False Negatives (yes→no):  {trad_cm['FN']}")

    if (trad_cm['TP'] + trad_cm['FP']) > 0:
        trad_precision = trad_cm['TP'] / (trad_cm['TP'] + trad_cm['FP']) * 100
        print(f"   Precision: {trad_precision:.2f}%")
    if (trad_cm['TP'] + trad_cm['FN']) > 0:
        trad_recall = trad_cm['TP'] / (trad_cm['TP'] + trad_cm['FN']) * 100
        print(f"   Recall: {trad_recall:.2f}%")

    print("\nSNN Model:")
    print(f"   True Positives (yes→yes):  {snn_cm['TP']}")
    print(f"   True Negatives (no→no):    {snn_cm['TN']}")
    print(f"   False Positives (no→yes):  {snn_cm['FP']}")
    print(f"   False Negatives (yes→no):  {snn_cm['FN']}")

    if (snn_cm['TP'] + snn_cm['FP']) > 0:
        snn_precision = snn_cm['TP'] / (snn_cm['TP'] + snn_cm['FP']) * 100
        print(f"   Precision: {snn_precision:.2f}%")
    if (snn_cm['TP'] + snn_cm['FN']) > 0:
        snn_recall = snn_cm['TP'] / (snn_cm['TP'] + snn_cm['FN']) * 100
        print(f"   Recall: {snn_recall:.2f}%")

    # Energy efficiency analysis
    print("\n⚡ ENERGY EFFICIENCY ANALYSIS:")
    print("-" * 50)
    energy_ratio = metrics['snn']['avg_energy'] / metrics['trad']['avg_energy']
    if energy_ratio < 1:
        print(f"💚 SNN is {1 / energy_ratio:.1f}x MORE energy efficient than Traditional model")
        print(f"   {metrics['snn']['avg_energy']:.2f} vs {metrics['trad']['avg_energy']:.2f} energy units")
    else:
        print(f"💛 Traditional model is {energy_ratio:.1f}x more energy efficient")

    print(f"🧠 Average SNN spike rate: {metrics['snn']['avg_spike_rate']:.4f} spikes/neuron/timestep")

    # Analysis and recommendation
    print("\n📈 PERFORMANCE ANALYSIS:")
    print("-" * 50)

    acc_diff = abs(metrics['trad']['accuracy'] - metrics['snn']['accuracy'])
    if metrics['trad']['accuracy'] > metrics['snn']['accuracy']:
        print(f"🎯 Traditional model is {acc_diff:.2f}% MORE ACCURATE")
    elif metrics['snn']['accuracy'] > metrics['trad']['accuracy']:
        print(f"🎯 SNN model is {acc_diff:.2f}% MORE ACCURATE")
    else:
        print(f"🎯 Both models have EQUAL accuracy")

    speed_diff = abs(metrics['trad']['avg_time'] - metrics['snn']['avg_time'])
    if metrics['trad']['avg_time'] < metrics['snn']['avg_time']:
        print(
            f"⚡ Traditional model is {speed_diff:.2f}ms FASTER ({speed_diff / metrics['snn']['avg_time'] * 100:.1f}% faster)")
    elif metrics['snn']['avg_time'] < metrics['trad']['avg_time']:
        print(
            f"⚡ SNN model is {speed_diff:.2f}ms FASTER ({speed_diff / metrics['trad']['avg_time'] * 100:.1f}% faster)")
    else:
        print(f"⚡ Both models have EQUAL speed")

    # Recommendation
    print("\n💡 FINAL RECOMMENDATION:")
    print("-" * 50)

    if metrics['trad']['accuracy'] > metrics['snn']['accuracy'] + 10:
        print("   ✅ Traditional model is SIGNIFICANTLY better for accuracy-critical applications")
        print("   📌 Use Traditional for production systems where accuracy is priority")
    elif metrics['snn']['accuracy'] > metrics['trad']['accuracy'] + 10:
        print("   ✅ SNN model is SIGNIFICANTLY better - great for neuromorphic hardware!")
        print("   📌 Use SNN for edge AI, low-power devices, or research applications")
        print(f"   📌 SNN provides {energy_ratio:.1f}x energy savings vs traditional")
    elif energy_ratio < 0.5:
        print("   ✅ SNN offers SIGNIFICANT energy savings with competitive accuracy")
        print("   📌 Ideal for battery-powered edge devices and neuromorphic computing")
    elif metrics['trad']['accuracy'] > metrics['snn']['accuracy']:
        print("   📌 Traditional model slightly better, but SNN is competitive")
        print("   📌 Consider deployment constraints (power, hardware) for final choice")
    else:
        print("   📌 SNN model is competitive with traditional ML")
        print("   📌 SNN offers advantages for spatiotemporal processing and hardware efficiency")


# ==============================
# MAIN APPLICATION
# ==============================
def main():
    print("=" * 70)
    print("🎯 VOICE COMMAND COMPARISON SYSTEM")
    print("Traditional (LR) vs Spiking Neural Network (SNN)")
    print("=" * 70)

    # Load models
    trad_model, encoder = load_traditional_model()
    snn_model, X_min, X_max, idx_to_label = load_snn_model()

    if trad_model is None:
        print("\n❌ Traditional model not found! Train it first.")
        return

    if snn_model is None:
        print("\n⚠️  SNN model not found! Only Traditional model available.")

    tracker = ComparisonTracker()

    while True:
        print("\n" + "=" * 70)
        print("📋 OPTIONS:")
        print("  1. 🎤 Record voice command (yes/no)")
        print("  2. 📁 Test with WAV file")
        print("  3. 📊 Show comparison report")
        print("  4. 🗑️  Reset all results")
        print("  5. 🚪 Exit")
        print("=" * 70)

        choice = input("\n👉 Choose (1-5): ").strip()

        if choice == '1':
            # Record from microphone
            true_label = input("What command will you say? (yes/no): ").strip().lower()
            if true_label not in ['yes', 'no']:
                print("❌ Invalid command! Use 'yes' or 'no'")
                continue

            input("\nPress Enter, then speak clearly...")
            audio = record_audio()

        elif choice == '2':
            # Test with file
            filepath = input("Enter WAV file path: ").strip()
            if not os.path.exists(filepath):
                print("❌ File not found!")
                continue

            try:
                audio, sr = librosa.load(filepath, sr=16000)
                true_label = input("Actual command? (yes/no): ").strip().lower()
                if true_label not in ['yes', 'no']:
                    print("❌ Invalid command!")
                    continue
            except Exception as e:
                print(f"❌ Error loading audio: {e}")
                continue

        elif choice == '3':
            # Show report
            metrics = tracker.get_metrics()
            if metrics:
                trad_cm, snn_cm = tracker.get_confusion_matrices()
                print_detailed_report(metrics, trad_cm, snn_cm)

                # Generate visualization if enough data
                if metrics['n'] > 1:
                    try:
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                        # Accuracy comparison
                        axes[0, 0].bar(['Traditional', 'SNN'],
                                       [metrics['trad']['accuracy'], metrics['snn']['accuracy']],
                                       color=['#2ecc71', '#e74c3c'])
                        axes[0, 0].set_ylabel('Accuracy (%)')
                        axes[0, 0].set_title('Model Accuracy Comparison')
                        axes[0, 0].set_ylim(0, 100)
                        for i, v in enumerate([metrics['trad']['accuracy'], metrics['snn']['accuracy']]):
                            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

                        # Time comparison
                        axes[0, 1].bar(['Traditional', 'SNN'],
                                       [metrics['trad']['avg_time'], metrics['snn']['avg_time']],
                                       color=['#2ecc71', '#e74c3c'])
                        axes[0, 1].set_ylabel('Time (ms)')
                        axes[0, 1].set_title('Inference Speed (lower is better)')
                        for i, v in enumerate([metrics['trad']['avg_time'], metrics['snn']['avg_time']]):
                            axes[0, 1].text(i, v + 0.5, f'{v:.1f}ms', ha='center', fontsize=12, fontweight='bold')

                        # Energy comparison
                        axes[1, 0].bar(['Traditional', 'SNN'],
                                       [metrics['trad']['avg_energy'], metrics['snn']['avg_energy']],
                                       color=['#2ecc71', '#e74c3c'])
                        axes[1, 0].set_ylabel('Energy (units)')
                        axes[1, 0].set_title('Energy Consumption (lower is better)')
                        for i, v in enumerate([metrics['trad']['avg_energy'], metrics['snn']['avg_energy']]):
                            axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')

                        # Spike rate (only for SNN)
                        axes[1, 1].bar(['SNN Spike Rate'], [metrics['snn']['avg_spike_rate']],
                                       color=['#3498db'])
                        axes[1, 1].set_ylabel('Spikes/Neuron/Timestep')
                        axes[1, 1].set_title('SNN Neural Activity (lower = more efficient)')
                        axes[1, 1].text(0, metrics['snn']['avg_spike_rate'] + 0.001,
                                        f'{metrics["snn"]["avg_spike_rate"]:.4f}',
                                        ha='center', fontsize=12, fontweight='bold')

                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"Could not generate plot: {e}")
            else:
                print("\n⚠️ No test results yet! Test some commands first.")
            continue

        elif choice == '4':
            tracker = ComparisonTracker()
            print("\n🗑️ All results reset!")
            continue

        elif choice == '5':
            print("\n👋 Exiting... Thank you for using the comparison system!")
            break

        else:
            print("❌ Invalid choice! Enter 1-5")
            continue

        # Extract features and predict
        features = extract_features(audio)

        # Traditional prediction
        trad_word, trad_conf, trad_time = predict_traditional(trad_model, encoder, features)
        trad_correct = trad_word == true_label
        trad_energy = len(features) * 2 * 10  # features * classes * ENERGY_PER_MAC

        # SNN prediction (if available)
        if snn_model:
            snn_word, snn_conf, snn_time, spike_rate, total_spikes, snn_energy, _ = predict_snn(
                snn_model, features, X_min, X_max, idx_to_label
            )
            snn_correct = snn_word == true_label
        else:
            snn_word, snn_conf, snn_time, spike_rate, snn_energy, snn_correct = "N/A", 0, 0, 0, 0, False
            trad_energy = 0

        # Store results
        tracker.add_result(true_label, trad_word, snn_word,
                           trad_conf, snn_conf,
                           trad_time, snn_time,
                           spike_rate, snn_energy, trad_energy)

        # Display
        print_results(true_label, trad_word, trad_conf, trad_time, trad_correct,
                      snn_word, snn_conf, snn_time, snn_correct,
                      spike_rate, snn_energy, trad_energy)

        # Show cumulative accuracy
        metrics = tracker.get_metrics()
        if metrics:
            print(f"\n📈 Cumulative Performance:")
            print(f"   Traditional: {metrics['trad']['accuracy']:.1f}% ({metrics['trad']['correct']}/{metrics['n']})")
            if snn_model:
                print(f"   SNN: {metrics['snn']['accuracy']:.1f}% ({metrics['snn']['correct']}/{metrics['n']})")
                print(f"   Avg Spike Rate: {metrics['snn']['avg_spike_rate']:.4f}")


if __name__ == "__main__":
    # Check for required packages
    missing_packages = []
    for package in ['sounddevice', 'tabulate', 'matplotlib', 'soundfile']:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        print("\nContinuing with available features...")

    main()