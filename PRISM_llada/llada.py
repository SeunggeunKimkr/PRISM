import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llada_utils import load_llada_modules
(LLaDASequentialBlock, ModelConfig, ActivationType, LayerNormType) = load_llada_modules()


# ------------------------------------------------------------
# additional sigmoid head
# ------------------------------------------------------------
class RemaskingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.proj1(h)
        h = self.act(h)
        h = self.proj2(h)
        return h

class RemaskingLLaDA(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()

        # define the architecture configuations
        self.backbone = backbone
        self.d_model = d_model
        self.remasking_head = RemaskingHead(d_model)

        # extract the core transformer blocks (backbone model is PEFT model)
        core = self.backbone.base_model.model.model.transformer
    
    def forward(self, input_ids: torch.LongTensor, **kwargs):
        # for eval loop, we drop the kwargs that are not needed
        for k in ("x0", "t", "masked_indices"):
            kwargs.pop(k, None)

        out = self.backbone(input_ids = input_ids, output_hidden_states = True, return_dict = True, **kwargs)
        
        # extract the last hidden state
        hidden_v = out.hidden_states[-1]
        remasking_conf = self.remasking_head(hidden_v)

        return {"logits": out.logits, "remasking_conf": remasking_conf}
    
    @property
    def device(self):
        return self.backbone.device

