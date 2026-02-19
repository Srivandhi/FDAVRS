import torch
import torch.nn as nn

class LocalAdapter:
    def __init__(self, model):
        self.model = model

    def adapt(self, inputs):
        """
        Iterative Adaptation: Forces the model to update Batch Norm stats
        by seeing the same noisy image multiple times (The 'Force-Feed' method).
        """
        # 1. SAVE STATE
        was_training = self.model.training
        
        # 2. OPEN EARS (Train Mode = ON)
        self.model.train() 
        
        # 3. TURBO LOOP (The Fix)
        # We repeat the forward pass 10 times.
        # This pushes the 'running_mean' and 'running_var' to match the noise.
        n_iterations = 10  
        
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = self.model(inputs)
            
        # 4. RESTORE STATE
        if not was_training:
            self.model.eval()