import torch
import torch.nn as nn
from torchcrf import CRF

class CRFOutputLayer(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, outputs, labels=None, mask=None):
        emissions = self.fc(outputs)

        if labels is not None:
            # CRF requires first token to be valid, so we create a modified mask
            # that ensures first token is always included
            if mask is None:
                mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                mask = mask.bool()
            
            # Ensure first position is always valid for CRF
            mask[:, 0] = True
            
            # Replace -100 with 0 (dummy label) to avoid index issues
            labels_crf = labels.clone()
            labels_crf[labels == -100] = 0
            
            # Calculate loss
            log_likelihood = self.crf(emissions, tags=labels_crf, mask=mask, reduction="mean")
            loss = -log_likelihood
            return {"logits": emissions, "loss": loss}
        else:
            if mask is None:
                mask = torch.ones(outputs.shape[:2], dtype=torch.bool, device=outputs.device)

            pred = self.crf.decode(emissions, mask=mask.bool()) # type: ignore
            return {"logits": emissions, "pred": pred}