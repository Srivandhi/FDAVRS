import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine

class UniversalDriftSDK:
    def __init__(self, model, feature_layer_name='avgpool'):
        self.model = model
        self.device = next(model.parameters()).device
        self.feature_layer_name = feature_layer_name
        
        self._current_features = None
        self.reference_embedding = None 
        
        self._attach_hook()

    def _attach_hook(self):
        layer_found = False
        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                module.register_forward_hook(self._hook_fn)
                layer_found = True
                break
        
        if not layer_found:
            print(f"[SDK Error] Layer '{self.feature_layer_name}' not found!")

    def _hook_fn(self, module, input, output):
        # Flatten features: (Batch, 512, 1, 1) -> (Batch, 512)
        self._current_features = output.detach().flatten(start_dim=1)

    def fit_baseline(self, dataloader, num_batches=10):
        print("[SDK] Calibration: Learning baseline patterns...")
        self.model.eval()
        embeddings_sum = None
        count = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches: break
                images = images.to(self.device)
                _ = self.model(images) # Trigger hook
                
                batch_features = self._current_features.cpu().numpy()
                
                if embeddings_sum is None:
                    embeddings_sum = np.sum(batch_features, axis=0)
                else:
                    embeddings_sum += np.sum(batch_features, axis=0)
                
                count += batch_features.shape[0]

        self.reference_embedding = embeddings_sum / count
        print(f"[SDK] Baseline Set. Reference Vector Size: {self.reference_embedding.shape}")

    def predict_metrics(self, logits):
        """Calculates metrics based on the current logits and captured features."""
        probs = F.softmax(logits, dim=1)
        
        # A: Confidence
        confidence, _ = torch.max(probs, dim=1)
        avg_confidence = confidence.mean().item()
        
        # B: Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        avg_entropy = entropy.mean().item()
        
        # C: Embedding Shift
        if self.reference_embedding is not None:
            current_avg = self._current_features.mean(dim=0).cpu().numpy()
            shift = cosine(self.reference_embedding, current_avg)
        else:
            shift = 0.0

        # D: Composite Score
        drift_score = (0.5 * avg_entropy) + (0.3 * shift) + (0.2 * (1 - avg_confidence))

        return {
            "score": drift_score,
            "entropy": avg_entropy,
            "shift": shift,
            "confidence": avg_confidence
        }