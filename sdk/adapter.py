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
        self.model_type = "yolo" if hasattr(model, 'names') else "resnet"

    def adapt(self, inputs):
        """
        Universal Test-Time Adaptation: Updates normalization layers locally.
        This is the 'adapt' method your core.py is looking for.
        """
        self.model.train() 
        # 5-10 iterations is usually enough to stabilize Batch Norm stats
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(inputs)
        self.model.eval()

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