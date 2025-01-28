import torch  
import yaml  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  
from models.reward_model import RewardModel  
from trainers.sft_trainer import SFTTrainer  
from trainers.reward_trainer import RewardTrainer  
from trainers.ppo_trainer import PPOTrainer  
from utils.data.data_utils import SFTDataset, RewardModelDataset  
from torch.utils.data import DataLoader  

def load_config():  
    with open("config/config.yaml", "r") as f:  
        return yaml.safe_load(f)  

def main():  
    # 加载配置  
    config = load_config()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    # 加载tokenizer  
    tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["base_model"])  
    tokenizer.pad_token = tokenizer.eos_token  
    
    # 1. SFT训练  
    print("Starting SFT training...")  
    sft_model = GPT2LMHeadModel.from_pretrained(config["model"]["base_model"])  
    sft_dataset = SFTDataset("data/sft_data.json", tokenizer, config["model"]["max_length"])  
    sft_dataloader = DataLoader(  
        sft_dataset,  
        batch_size=config["sft"]["batch_size"],  
        shuffle=True  
    )  
    
    sft_trainer = SFTTrainer(sft_model, tokenizer, config, device)  
    sft_trainer.train(sft_dataloader)  
    
    # 2. 奖励模型训练  
    print("Starting Reward Model training...")