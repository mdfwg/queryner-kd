# src/models/base_model.py
from torch import nn

class BaseNERModel(nn.Module):
    """
    Abstract base class untuk model NER:
    - punya atribut num_labels
    - mendukung dua mode: base (softmax) dan crf
    """
    def __init__(self, num_labels, use_crf=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf

    def forward(self, input_ids, attention_mask, labels=None):
        raise NotImplementedError("Forward harus diimplementasikan di subclass.")