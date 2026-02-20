import torch
import torch.nn as nn
import os
import json
import datetime

class LocalAdapter:
    def __init__(self, model):
        self.model = model
        # The Local Vault stores the knowledge packages for the server
        self.vault_path = os.path.normpath(os.path.join(os.getcwd(), "local_knowledge_vault"))
        os.makedirs(self.vault_path, exist_ok=True)
        # Identify model type for FedAvg categorization
        # YOLO models usually have a 'names' attribute or 'model.names' if wrapped
        is_yolo = hasattr(model, 'names') or (hasattr(model, 'model') and hasattr(model.model, 'names'))
        self.model_type = "yolo" if is_yolo else "resnet"
        # State tracking to prevent knowledge spam
        self.has_discovered = False

    def adapt(self, inputs):
        """Universal Test-Time Adaptation: Updates normalization layers locally."""
        self.model.train() 
        # 5 iterations is original baseline for Client Alpha
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(inputs)
        self.model.eval()

    def synchronize_knowledge(self, images, metrics, brain_action, monitor, low_thresh):
        """
        Policy Engine: Decides when to capture and store domain knowledge.
        Aligns with FL Architecture: 
        - High Reliable -> Capture info for FL Server
        - Low Reliable -> Local Adaptation Layer
        """
        if self.has_discovered:
            return brain_action # One save per session to prevent spam
            
        drift_score = metrics.get('score', 0)
        
        # Scenario A: Local Adaptation triggered (Decision: unstable/recoverable)
        if brain_action == "LOCAL_ADAPTATION":
            self.adapt(images)
            with torch.no_grad():
                logits = self.model(images)
            post_metrics = monitor.predict_metrics(logits)
            
            # If successfully adapted, save to vault for FL Server path
            if post_metrics['score'] <= low_thresh:
                self.save_knowledge_package(post_metrics['score'], metrics)
                self.has_discovered = True
                return "KNOWLEDGE_BRIDGE_CREATED"
            return "LOCAL_ADAPTATION"
            
        # Scenario B: High Reliable Model (Decision: IDLE)
        # Even if IDLE, if we detect semantic shift (e.g. Fog), we capture knowledge for FL Server
        # For YOLO, we capture if score > 0.06 (baseline matches clean distribution)
        elif (self.model_type == "yolo" and 
              brain_action == "IDLE" and 
              drift_score > 0.06): 
            
            # Capturing weights for the 'High Reliable' path in the FL architecture
            self.adapt(images)
            self.save_knowledge_package(drift_score, metrics)
            self.has_discovered = True
            return "PASSIVE_DISCOVERY"
            
        return brain_action

    def save_knowledge_package(self, final_score, initial_metrics, n_samples=32):
        """
        Saves the successful adaptation weights and drift signature.
        This enables the 'Knowledge Bridge' to the Federated Server.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Capture the solution (Normalization weights only)
        state = {k: v.cpu() for k, v in self.model.state_dict().items() 
                 if 'norm' in k.lower() or 'bn' in k.lower()}
        
        weight_file = f"weights_{timestamp}.pt"
        torch.save(state, os.path.join(self.vault_path, weight_file))

        # 2. Capture the metadata for FedAvg learning
        package = {
            "signature": {
                "entropy": float(initial_metrics.get('entropy', 0)),
                "shift": float(initial_metrics.get('shift', 0)),
                "sample_count": n_samples # Crucial weighting factor for FedAvg
            },
            "solution_ref": weight_file,
            "final_reliability": float(final_score),
            "model_type": self.model_type
        }
        
        json_path = os.path.join(self.vault_path, f"pack_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(package, f, indent=4)
        
        print(f" [SDK] Knowledge Bridge Saved: {weight_file}")