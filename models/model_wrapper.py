import torch
import torch.nn as nn
import torch.nn.functional as F

class DITWrapper(nn.Module):
    def __init__(self, model: nn.Module, time_conditioning: bool = False, is_adapter: bool = False):
        super().__init__()
        self.model = model
        self.time_conditioning = time_conditioning
        self.is_adapter = is_adapter
        
    def _process_sigma(self, sigma):
        '''
        pre-process sigma (time-embedding) by replacing it with 0 if time_conditioning is False
        '''
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def get_output_and_features(self, x: torch.Tensor, sigma: torch.Tensor, **extras):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits, hidden_states, cond = self.model.get_hidden_states(x, sigma)
        return logits, hidden_states, cond
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **extras):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.model(x, sigma)
        if self.is_adapter:
            return logits.squeeze(-1)
        return logits

    def get_score(self, x: torch.Tensor, sigma: torch.Tensor, **extras):
        assert self.is_adapter
        return self.forward(x, sigma).sigmoid()
        

class FeatureDITWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, **extras):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.model(x, cond)
        return logits.squeeze(-1)

    def get_score(self, x: torch.Tensor, cond: torch.Tensor, **extras):
        return self.forward(x, cond).sigmoid()
            