# src/models/crf_layer.py
import torch
from torch import nn
from torchcrf import CRF

class CRFOutputLayer(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, outputs, labels=None, mask=None):
        emissions = self.fc(outputs)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            pred = self.crf.decode(emissions, mask=mask)
            return {"loss": loss, "pred": pred}
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return {"pred": pred}