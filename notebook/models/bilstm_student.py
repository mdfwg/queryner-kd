import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .crf_layer import CRFOutputLayer
from .base_model import BaseNERModel

class BiLSTMStudent(BaseNERModel):
    def __init__(
            self, 
            num_labels=35, 
            use_crf=False,
            model_name_for_vocab = 'bert-base-uncased',
            emb_dim = 300,
            lstm_hidden = 300,
            label_info = None,
            pad_token_id = 0
        ):
        super().__init__(num_labels, use_crf)
        self.use_crf = use_crf
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(model_name_for_vocab)
        vocab_size = self.config.vocab_size
        pad_token_id = self.config.pad_token_id

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        if self.use_crf:
            self.crf_output = CRFOutputLayer(hidden_dim=lstm_hidden * 2, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)
        outputs, _ = self.lstm(emb)
        sequence_output = outputs

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
