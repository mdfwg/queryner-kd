import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .crf_layer import CRFOutputLayer
from .base_model import BaseNERModel

class TinyBERTStudent(BaseNERModel):
    def __init__(self, model_name="huawei-noah/TinyBERT_General_4L_312D", label_info=None, use_crf=False):
        self.use_crf = use_crf
        self.num_labels = label_info["num_labels"] # type: ignore
        super().__init__(num_labels=self.num_labels, use_crf=self.use_crf)

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=label_info["num_labels"], # type: ignore
            id2label=label_info["id2label"], # type: ignore
            label2id=label_info["label2id"] # type: ignore
        )

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)

        if self.use_crf:
            self.crf_output = CRFOutputLayer(self.config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        if self.use_crf:
            mask = attention_mask.bool()
            result = self.crf_output(sequence_output, labels=labels, mask=mask)
            return result
        else:
            logits = self.classifier(sequence_output)
            if labels is not None:
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                return {"logits": logits, "loss": loss}
            else:
                pred = logits.argmax(dim=-1)
                return {"logits": logits, "pred": pred}