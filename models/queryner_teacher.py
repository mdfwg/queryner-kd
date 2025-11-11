# src/models/queryner_teacher.py
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from .crf_layer import CRFOutputLayer
from .base_model import BaseNERModel

class QueryNERTeacher(BaseNERModel):
    def __init__(self, model_name, label_info, use_crf=False):
        super().__init__(num_labels=label_info["num_labels"], use_crf=use_crf)

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=label_info["num_labels"],
            id2label=label_info["id2label"],
            label2id=label_info["label2id"]
        )

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)

        if use_crf:
            self.output = CRFOutputLayer(self.config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        if self.use_crf:
            mask = attention_mask.bool()
            result = self.output(sequence_output, labels=labels, mask=mask)
            return result
        else:
            logits = self.classifier(sequence_output)
            if labels is not None:
                loss = self.loss_fn(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
                return {"loss": loss, "logits": logits}
            else:
                return {"logits": logits}