import torch
from torch.utils.data import DataLoader
from .ner_dataset import NERDataset
from transformers import AutoTokenizer, AutoConfig

def load_label_info(model_name):
    config = AutoConfig.from_pretrained(model_name)
    id2label = config.id2label
    label2id = config.label2id
    num_labels = config.num_labels

    label_info = {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": num_labels
    }

    return label_info

def ner_collate_fn(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    tokens = [item["tokens"] for item in batch]
    ner_tags = [item["ner_tags"] for item in batch]
    word_ids = [item["word_ids"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "word_ids": word_ids
    }

def create_dataloaders(
        train_path, val_path, test_path,
        model_name,
        batch_size=32,
        max_length=128,
        collate_fn=ner_collate_fn
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NERDataset(train_path, tokenizer, max_length=max_length)
    val_dataset = NERDataset(val_path, tokenizer, max_length=max_length)
    test_dataset = NERDataset(test_path, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)

    return tokenizer, train_loader, val_loader, test_loader