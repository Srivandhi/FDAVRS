import torch
import cv2
import numpy as np
import os, glob, time, sys
from ultralytics import YOLO
import albumentations as A
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from sdk.core import FDAVRS 

DRIFT_START_ROUND = 4
TOTAL_ROUNDS = 8
QUICK_DEMO = True # Set to True to run only 15 rounds
if QUICK_DEMO:
    TOTAL_ROUNDS = 15
    DRIFT_START_ROUND = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "yolo_data", "coco128", "images", "train2017"))
SMOOTHING_WINDOW = 5

drift_pipeline = A.Compose([
    A.RandomFog(
        fog_coef_range=(0.9 , 0.95),
        alpha_coef=0.1,
        p=1.0
    )
])

class YoloConnector(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model 
        
    def forward(self, x):
        result = self.model(x)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

class ClientBetaYOLO:
    def __init__(self):
        print(f"[Init] Loading YOLO11n on {DEVICE}...")
        self.yolo_wrapper = YOLO("yolo11n.pt") 
        
        self.image_pool = glob.glob(os.path.join(IMG_FOLDER, "*.jpg"))
        if not self.image_pool:
            print(f"[Error] No images found! Run setup_yolo_data.py first.")
            exit()
            
        self.client_model = YoloConnector(self.yolo_wrapper)
        self.sdk = FDAVRS(self.client_model, feature_layer='model.model.9', threshold=0.15)
        
        print("[SDK] Calibrating Baseline...")
        self.calibrate_sdk()
        
        self.conf_history = []
        self.smoothed_history = []
        self.score_history = []

    def preprocess_for_sdk(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(DEVICE)

    def calibrate_sdk(self):
        dummy_loader = []
        for i in range(5):
            img_path = self.image_pool[i]
            frame = cv2.resize(cv2.imread(img_path), (640, 640))
            tensor_img = self.preprocess_for_sdk(frame)
            dummy_loader.append((tensor_img, None))
        self.sdk.fit_baseline(dummy_loader)

    def apply_drift(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = drift_pipeline(image=frame_rgb)["image"]
        return cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)

    def run_simulation(self):
        print(f"\n[Simulation] Starting YOLO Reliability Demo with SDK...")
        print(f"{'Round':<6} | {'Status':<12} | {'Action':<16} | {'Raw Conf':<8} | {'Smooth':<8}")
        print("-" * 65)

        for round_id in range(1, TOTAL_ROUNDS + 1):
            img_path = self.image_pool[round_id % len(self.image_pool)]
            frame = cv2.resize(cv2.imread(img_path), (640, 640))
            status = "CLEAN"

            if round_id >= DRIFT_START_ROUND:
                frame = self.apply_drift(frame)
                status = "FOG_DRIFT"

            tensor_img = self.preprocess_for_sdk(frame)
            
            # Reduce repetition from 16 to 8 for speed
            batch_input = tensor_img.repeat(8, 1, 1, 1)
            
            _ = self.sdk.predict(batch_input)

            results = self.yolo_wrapper(frame, device=DEVICE, verbose=False)
            
            confs = results[0].boxes.conf.cpu().numpy()
            avg_conf = np.mean(confs) if len(confs) > 0 else 0.0
            
            score = self.sdk.last_metrics.get('score', 0)
            action = self.sdk.last_action
            
            self.conf_history.append(avg_conf)
            self.score_history.append(score)
            
            if len(self.conf_history) >= SMOOTHING_WINDOW:
                smoothed_conf = np.mean(self.conf_history[-SMOOTHING_WINDOW:])
            else:
                smoothed_conf = np.mean(self.conf_history)
            self.smoothed_history.append(smoothed_conf)

            annotated_frame = results[0].plot() 
            color = (0, 255, 0) if status == "CLEAN" else (0, 0, 255)
            if action == "LOCAL_ADAPTATION": color = (255, 165, 0) 
            
            cv2.putText(annotated_frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated_frame, f"ACTION: {action}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated_frame, f"SMOOTH CONF: {smoothed_conf:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("Client Beta - YOLO + FDAVRS", annotated_frame)
            
            print(f"{round_id:<6} | {status:<12} | {action:<16} | {avg_conf:.3f}    | {smoothed_conf:.3f}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

        cv2.destroyAllWindows()
        self.plot_report()

    def plot_report(self):
        rounds = range(1, TOTAL_ROUNDS + 1)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(rounds, self.conf_history, color='orange', alpha=0.3, label='Raw Confidence')
        ax1.plot(rounds, self.smoothed_history, color='darkorange', linewidth=3, label=f'Trend (n={SMOOTHING_WINDOW})')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Detection Confidence Score', fontsize=12, color='darkorange')
        ax1.tick_params(axis='y', labelcolor='darkorange')
        ax1.set_ylim(0, 1.0)
        
        ax2 = ax1.twinx()
        ax2.plot(rounds, self.score_history, color='red', linestyle='--', alpha=0.6, label='SDK Drift Score')
        ax2.set_ylabel('Drift Score', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1.0)
        
        plt.axvline(x=DRIFT_START_ROUND, color='black', linestyle=':', linewidth=2, label='Fog Injected')
        plt.title('Client Beta: YOLOv11 Drift Adaptation with FDAVRS', fontsize=14)
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Save to output folder
        output_dir = os.path.join(os.path.dirname(BASE_DIR), "output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "yolo_sdk_final_report.png")
        
        plt.savefig(save_path)
        print(f"\n[Graph] Report saved as {save_path}")
        plt.show()

if __name__ == "__main__":
    client = ClientBetaYOLO()
    client.run_simulation()