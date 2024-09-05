import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrOCRProcessor
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.max_seq_len = config.max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        # Check if 'text' is the correct column name, or if it should be something else, e.g., 'transcription'
        text = self.df['label'][idx]
        input_ids = self.tokenizer(str(text), return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_len)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        return {
            "pixel_values":    pixel_values[0],
            "input_ids":   input_ids["input_ids"][0],
            "att_mask":     input_ids["attention_mask"][0]
        }
