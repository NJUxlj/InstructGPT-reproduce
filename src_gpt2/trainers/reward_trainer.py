import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import wandb  
from tqdm import tqdm  

class RewardTrainer:  
    def __init__(  
        self,  
        model,  
        tokenizer,  
        config,  
        device  
    ):  
        self.model = model.to(device)  
        self.tokenizer = tokenizer  
        self.config = config  
        self.device = device  
        
        self.optimizer = optim.AdamW(  
            self.model.parameters(),  
            lr=config["reward_model"]["learning_rate"]  
        )  
        
    def train_step(self, batch):  
        better_rewards = self.model(  
            input_ids=batch["better_input_ids"].to(self.device),  
            attention_mask=batch["better_attention_mask"].to(self.device)  
        )  
        
        worse_rewards = self.model(  
            input_ids=batch["worse_input_ids"].to(self.device),  
            attention_mask=batch["worse_attention_mask"].to(self.device)  
        )  
        
        loss = -torch.log(torch.sigmoid(better_rewards - worse_rewards)).mean()  
        
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  
        
        return {  
            "loss": loss.item(),  
            "reward_diff": (better_rewards - worse_rewards).mean().item()  
        }  
    
    def train(self, train_dataloader):  
        self.model.train()  
        wandb.init(project="instructgpt-reward-model")  
        
        for epoch in range(self.config["reward_model"]["num_epochs"]):  
            total_loss = 0  
            total_reward_diff = 0  
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")  
            for batch in progress_bar:  
                stats = self.train_step(batch)  
                total_loss += stats["loss"]  
                total_reward_diff += stats["reward_diff"]  
                
                wandb.log(stats)  
                progress_bar.set_postfix(loss=stats["loss"])  
            
            avg_loss = total_loss / len(train_dataloader)  
            avg_reward_diff = total_reward_diff / len(train_dataloader)  
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Avg Reward Diff = {avg_reward_diff:.4f}")