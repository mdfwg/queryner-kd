import json
import torch
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_pad_id=-100, max_length=128):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)["examples"]
        self.data = raw
        self.tokenizer = tokenizer
        self.label_pad_id = label_pad_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        ner_tags = self.data[idx]["ner_tags"]

        # buat encoding untuk tokens 
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        # align labels dengan tokens yang sudah diencoding (jadi kepotong2 sesuai tokenization)
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(self.label_pad_id)
            elif word_idx != previous_word_idx:
                aligned_labels.append(ner_tags[word_idx])
            else:
                aligned_labels.append(self.label_pad_id)
            previous_word_idx = word_idx
        
        item = {
            "tokens": tokens,
            "ner_tags": ner_tags,
            "word_ids": word_ids,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

        return item