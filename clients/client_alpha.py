import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# --- DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
DRIFT_START_ROUND = 50   
TOTAL_ROUNDS = 100       
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the corruption type here (e.g., 'fog', 'snow', 'motion_blur', 'brightness')
CORRUPTION_TYPE = "fog" 

# Paths
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "resnet18_cifar10_trained.pth"))
CORRUPTION_NPY_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "CIFAR-10-C", f"{CORRUPTION_TYPE}.npy"))
LABELS_NPY_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "CIFAR-10-C", "labels.npy"))

# Visual Settings
ENABLE_LIVE_LOGS = True  
LOG_DELAY = 0.05         
SAVE_GRAPH = True        

class ClientAlphaUnified:
    def __init__(self):
        print(f"[Init] Corruption Mode: {CORRUPTION_TYPE.upper()}")
        print(f"[Init] Loading Model on {DEVICE}...")
        
        # 1. Load Model Structure & Weights
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        
        if not os.path.exists(MODEL_PATH):
            print(f"[Error] Model not found at: {MODEL_PATH}")
            sys.exit(1)
            
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()

        # 2. DATA LOADING: CLEAN CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        clean_root = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
        clean_dataset = torchvision.datasets.CIFAR10(root=clean_root, train=False, download=True, transform=transform)
        self.clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.clean_iter = iter(self.clean_loader)

        # 3. DATA LOADING: CORRUPTED DATA (CIFAR-10-C)
        if not os.path.exists(CORRUPTION_NPY_PATH):
            print(f"[Error] Corruption file NOT found at: {CORRUPTION_NPY_PATH}")
            sys.exit(1)

        # Load .npy files
        all_drift_images = np.load(CORRUPTION_NPY_PATH)  
        all_labels = np.load(LABELS_NPY_PATH)      

        # Severity 5: Select the last 10,000 images in the 50,000 image block
        self.drift_images = all_drift_images[40000:]
        self.drift_labels = all_labels[40000:]
        self.drift_idx = 0

        self.accuracy_history = []

    def preprocess_npy(self, batch_images):
        """Converts raw numpy images to normalized tensors for ResNet"""
        batch_images = batch_images.astype(np.float32) / 255.0
        batch_images = (batch_images - 0.5) / 0.5
        tensor = torch.tensor(batch_images).permute(0, 3, 1, 2)
        return tensor

    def get_batch(self, round_id):
        if round_id < DRIFT_START_ROUND:
            # Use Clean CIFAR-10
            try:
                images, labels = next(self.clean_iter)
            except StopIteration:
                self.clean_iter = iter(self.clean_loader)
                images, labels = next(self.clean_iter)
            status = "CLEAN"
        else:
            # Use Corrupted Data
            if self.drift_idx + BATCH_SIZE > len(self.drift_images):
                self.drift_idx = 0 
            
            img_batch = self.drift_images[self.drift_idx : self.drift_idx + BATCH_SIZE]
            lbl_batch = self.drift_labels[self.drift_idx : self.drift_idx + BATCH_SIZE]
            
            images = self.preprocess_npy(img_batch)
            labels = torch.tensor(lbl_batch).long()
            
            self.drift_idx += BATCH_SIZE
            status = f"DRIFT_{CORRUPTION_TYPE.upper()}"
            
        return images.to(DEVICE), labels.to(DEVICE), status

    def run_simulation(self):
        print(f"\n[Simulation] Starting {TOTAL_ROUNDS} Rounds...")
        if ENABLE_LIVE_LOGS:
            print(f"{'Round':<8} | {'Status':<15} | {'Accuracy':<10} | {'Prediction'}")
            print("-" * 60)

        for round_id in range(1, TOTAL_ROUNDS + 1):
            images, labels, status = self.get_batch(round_id)

            with torch.no_grad():
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

            correct = (preds == labels).sum().item()
            accuracy = (correct / BATCH_SIZE) * 100
            self.accuracy_history.append(accuracy)

            if ENABLE_LIVE_LOGS:
                pred_ex = preds[0].item()
                print(f"{round_id:<8} | {status:<15} | {accuracy:.1f}%     | Class {pred_ex}")
                time.sleep(LOG_DELAY)

        if SAVE_GRAPH:
            self.plot_results()

    def plot_results(self):
        print("\n[Graph] Rendering reliability report...")
        rounds = range(1, TOTAL_ROUNDS + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.accuracy_history, label='ResNet-18 Accuracy', color='#1f77b4', linewidth=2)
        plt.axvline(x=DRIFT_START_ROUND, color='red', linestyle='--', label=f'Drift Starts ({CORRUPTION_TYPE})')
        
        plt.fill_between(rounds, self.accuracy_history, color='skyblue', alpha=0.3)
        plt.title(f'Reliability Decay: Clean Data vs. {CORRUPTION_TYPE.capitalize()} Drift', fontsize=14)
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(loc='lower left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(0, 105)
        
        filename = f"{CORRUPTION_TYPE}_drift_report.png"
        plt.savefig(filename)
        print(f"[Graph] Saved to: {os.path.abspath(filename)}")
        plt.show()

if __name__ == "__main__":
    client = ClientAlphaUnified()
    client.run_simulation()