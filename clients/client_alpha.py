import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import random

# --- 1. SEED SETUP ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Setup] Random Seed set to {seed}. Results are now reproducible.")

set_seed(42)

# --- 2. PATH & SDK SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define ROOT_DIR globally so it's accessible everywhere
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

sys.path.append(ROOT_DIR)
from sdk.core import FDAVRS

# --- 3. CONFIGURATION ---
DRIFT_START_ROUND = 50   
TOTAL_ROUNDS = 100       
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORRUPTION_TYPE = "fog" 

MODEL_PATH = os.path.join(ROOT_DIR, "resnet18_cifar10_trained.pth")
CORRUPTION_NPY_PATH = os.path.join(ROOT_DIR, "data", "CIFAR-10-C", f"{CORRUPTION_TYPE}.npy")
LABELS_NPY_PATH = os.path.join(ROOT_DIR, "data", "CIFAR-10-C", "labels.npy")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

class ClientAlphaUnified:
    def __init__(self):
        print(f"[Init] Corruption Mode: {CORRUPTION_TYPE.upper()}")
        
        # Load Model
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        
        if not os.path.exists(MODEL_PATH):
            print(f"[Error] Model not found at: {MODEL_PATH}")
            sys.exit(1)
            
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model = self.model.to(DEVICE).eval()

        # Initialize SDK
        self.sdk = FDAVRS(self.model, feature_layer='avgpool')

        # Load Clean Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        clean_root = os.path.join(ROOT_DIR, "data")
        clean_dataset = torchvision.datasets.CIFAR10(root=clean_root, train=False, download=True, transform=transform)
        self.clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.clean_iter = iter(self.clean_loader)

        # Calibrate SDK
        self.sdk.fit_baseline(self.clean_loader)

        # Load Drift Data
        all_drift = np.load(CORRUPTION_NPY_PATH)
        all_labels = np.load(LABELS_NPY_PATH)
        self.drift_images = all_drift[40000:]
        self.drift_labels = all_labels[40000:]
        self.drift_idx = 0

        self.history = {'acc': [], 'score': []}

    def preprocess_npy(self, batch_images):
        batch_images = batch_images.astype(np.float32) / 255.0
        batch_images = (batch_images - 0.5) / 0.5
        return torch.tensor(batch_images).permute(0, 3, 1, 2)

    def get_batch(self, round_id):
        if round_id < DRIFT_START_ROUND:
            try:
                images, labels = next(self.clean_iter)
            except StopIteration:
                self.clean_iter = iter(self.clean_loader)
                images, labels = next(self.clean_iter)
            status = "CLEAN"
        else:
            if self.drift_idx + BATCH_SIZE > len(self.drift_images):
                self.drift_idx = 0
            
            img_batch = self.drift_images[self.drift_idx : self.drift_idx + BATCH_SIZE]
            lbl_batch = self.drift_labels[self.drift_idx : self.drift_idx + BATCH_SIZE]
            
            images = self.preprocess_npy(img_batch)
            labels = torch.tensor(lbl_batch).long()
            self.drift_idx += BATCH_SIZE
            status = "DRIFT"
            
        return images.to(DEVICE), labels.to(DEVICE), status

    def run_simulation(self):
        print(f"\n[Simulation] Starting {TOTAL_ROUNDS} Rounds...")
        print(f"{'Round':<6} | {'Status':<8} | {'Score':<6} | {'Action':<18} | {'Acc':<5}")
        print("-" * 65)

        for round_id in range(1, TOTAL_ROUNDS + 1):
            images, labels, status = self.get_batch(round_id)
            
            # Predict via SDK
            outputs = self.sdk.predict(images)
            
            drift_score = self.sdk.last_metrics['score']
            action = self.sdk.last_action
            
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).sum().item() / BATCH_SIZE * 100
            
            self.history['acc'].append(acc)
            self.history['score'].append(drift_score)

            print(f"{round_id:<6} | {status:<8} | {drift_score:.2f}   | {action:<18} | {acc:.1f}%")
            if round_id % 5 == 0: time.sleep(0.01) 

        self.plot_results()

    def plot_results(self):
        rounds = range(1, TOTAL_ROUNDS + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)', color='tab:blue')
        ax1.plot(rounds, self.history['acc'], color='tab:blue', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 100)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Drift Score', color='tab:red')
        ax2.plot(rounds, self.history['score'], color='tab:red', linestyle='--', label='Drift Score')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, 1.2)
        
        plt.title(f"FDAVRS SDK Performance: {CORRUPTION_TYPE.upper()} Adaptation")
        fig.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"{CORRUPTION_TYPE}_sdk_report.png")
        
        plt.savefig(save_path)
        print(f"\n[Graph] Report safely stored in the output folder: {save_path}")
        plt.show()

if __name__ == "__main__":
    client = ClientAlphaUnified()
    client.run_simulation()