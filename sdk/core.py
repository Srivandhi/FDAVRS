from .wrapper import UniversalDriftSDK
from .decision import DecisionMaker
from .adapter import LocalAdapter
import torch

class FDAVRS:
    # 1. Add 'threshold' here (Default is 0.3 for ResNet)
    def __init__(self, client_model, feature_layer='avgpool', threshold=0.3): 
        self.model = client_model
        
        # Initialize Sub-Systems
        self.monitor = UniversalDriftSDK(client_model, feature_layer_name=feature_layer)
        
        # 2. PASS THE THRESHOLD DOWN TO THE BRAIN
        # This is the missing link!
        self.brain = DecisionMaker(
            high_drift_threshold=0.8, 
            low_drift_threshold=threshold  # <--- CRITICAL FIX
        )
        
        self.adapter = LocalAdapter(client_model)
        
        # Logs
        self.last_action = "IDLE"
        self.last_metrics = {}

    def fit_baseline(self, dataloader):
        self.monitor.fit_baseline(dataloader)

    def predict(self, images):
        """
        The Main Loop: Monitor -> Decide -> Synchronize Knowledge -> Predict
        """
        self.model.eval()
        
        # 1. Initial Inference (to get metrics)
        with torch.no_grad():
            logits = self.model(images)
        
        # 2. Get Metrics & Decide
        metrics = self.monitor.predict_metrics(logits)
        self.last_metrics = metrics
        
        brain_action = self.brain.decide(metrics)
        self.last_action = brain_action # Update status first so it's visible in terminal
        
        # 3. Synchronize Knowledge: Comment the line below to show accuracy WITHOUT adaptation
        self.last_action = self.adapter.synchronize_knowledge(
            images, metrics, brain_action, self.monitor, self.brain.low_thresh
        )
        # pass
        # 4. If adaptation occurred, perform one final forward pass for the user's prediction
        if self.last_action in ["KNOWLEDGE_BRIDGE_CREATED", "PASSIVE_DISCOVERY"]:
            with torch.no_grad():
                logits = self.model(images)
        
        return logits