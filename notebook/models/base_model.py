import torch.nn as nn

class BaseNERModel(nn.Module):
    def __init__(self, num_labels, use_crf=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf

    def forward(self, input_ids, attention_mask, labels=None):
        raise NotImplementedError("Forward method must be implemented in subclass.")