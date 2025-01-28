import json  
from typing import List, Dict, Tuple  
from torch.utils.data import Dataset  

class SFTDataset(Dataset):  
    def __init__(self, data_path: str, tokenizer, max_length: int):  
        self.data = self._load_data(data_path)  
        self.tokenizer = tokenizer  
        self.max_length = max_length  

    def _load_data(self, data_path: str) -> List[Dict]:  
        with open(data_path, 'r', encoding='utf-8') as f:  
            return json.load(f)  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx: int):  
        item = self.data[idx]  
        prompt = item['prompt']  
        response = item['response']  
        
        # 构造输入格式：<|prompter|>prompt<|assistant|>response  
        full_text = f"<|prompter|>{prompt}<|assistant|>{response}"  
        
        encodings = self.tokenizer(  
            full_text,  
            max_length=self.max_length,  
            padding="max_length",  
            truncation=True,  
            return_tensors="pt"  
        )  
        
        return {  
            "input_ids": encodings["input_ids"].squeeze(),  
            "attention_mask": encodings["attention_mask"].squeeze(),  
            "labels": encodings["input_ids"].squeeze()  
        }  

class RewardModelDataset(Dataset):  
    def __init__(self, data_path: str, tokenizer, max_length: int):  
        self.data = self._load_data(data_path)  
        self.tokenizer = tokenizer  
        self.max_length = max_length  

    def _load_data(self, data_path: str) -> List[Dict]:  
        with open(data_path, 'r', encoding='utf-8') as f:  
            return json.load(f)  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx: int):  
        item = self.data[idx]  
        prompt = item['prompt']  
        better_response = item['better_response']  
        worse_response = item['worse_response']  
        
        better_text = f"<|prompter|>{prompt}<|assistant|>{better_response}"  
        worse_text = f"<|prompter|>{prompt}<|assistant|>{worse_response}"  
        
        better_encodings = self.tokenizer(  
            better_text,  
            max_length=self.max_length,  
            padding="max_length",  
            truncation=True,  
            return_tensors="pt"  
        )  
        
        worse_encodings = self.tokenizer(  
            worse_text,  
            max_length=self.max_length,  
            padding="max_length",  
            truncation=True,  
            return_tensors="pt"  
        )  
        
        return {  
            "better_input_ids": better_encodings["input_ids"].squeeze(),  
            "better_attention_mask": better_encodings["attention_mask"].squeeze(),  
            "worse_input_ids": worse_encodings["input_ids"].squeeze(),  
            "worse_attention_mask": worse_encodings["attention_mask"].squeeze(),  
        }