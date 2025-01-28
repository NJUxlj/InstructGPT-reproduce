import torch  
import torch.nn as nn  
import torch.optim as optim  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  
from torch.distributions import Categorical  
import numpy as np  
from typing import List, Tuple, Dict  
import wandb  # 用于训练监控  

class PPOTrainer:  
    def __init__(  
        self,  
        model_name: str = "gpt2",  
        lr: float = 1e-5,  
        beta: float = 0.2,  # KL散度系数  
        gamma: float = 0.1,  # 预训练目标系数  
        batch_size: int = 8,  
        max_length: int = 128,  
    ):  
        # 初始化模型和tokenizer  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        
        # 当前策略模型（用于优化）  
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)  
        # 参考模型（用于计算KL散度）  
        self.ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)  
        self.ref_model.eval()  # 固定参考模型  
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # 优化器  
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  
        
        # 超参数  
        self.beta = beta  
        self.gamma = gamma  
        self.batch_size = batch_size  
        self.max_length = max_length  

    def compute_rewards(self, texts: List[str]) -> torch.Tensor:  
        """  
        计算奖励值（这里使用一个简单的奖励模型示例）  
        实际应用中应该使用训练好的奖励模型  
        """  
        # 示例奖励计算  
        rewards = []  
        for text in texts:  
            # 这里用一个简单的启发式规则作为示例  
            # 实际应该使用训练好的奖励模型  
            reward = min(len(text.split()) / 10, 1.0)  # 简单示例：基于词数的奖励  
            rewards.append(reward)  
        return torch.tensor(rewards, device=self.device)  

    def compute_kl_divergence(  
        self,  
        current_logits: torch.Tensor,  
        ref_logits: torch.Tensor,  
        attention_mask: torch.Tensor  
    ) -> torch.Tensor:  
        """计算KL散度"""  
        current_probs = torch.softmax(current_logits, dim=-1)  
        ref_probs = torch.softmax(ref_logits, dim=-1)  
        
        kl_div = torch.sum(  
            current_probs * (torch.log(current_probs + 1e-10) - torch.log(ref_probs + 1e-10)),  
            dim=-1  
        )  
        # 使用attention mask只计算有效token的KL散度  
        kl_div = torch.sum(kl_div * attention_mask) / torch.sum(attention_mask)  
        return kl_div  

    def pretrain_loss(  
        self,  
        logits: torch.Tensor,  
        labels: torch.Tensor,  
        attention_mask: torch.Tensor  
    ) -> torch.Tensor:  
        """计算预训练损失（语言模型损失）"""  
        shift_logits = logits[..., :-1, :].contiguous()  
        shift_labels = labels[..., 1:].contiguous()  
        shift_mask = attention_mask[..., 1:].contiguous()  
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')  
        loss = loss_fct(  
            shift_logits.view(-1, shift_logits.size(-1)),  
            shift_labels.view(-1)  
        )  
        # 应用mask并计算平均损失  
        loss = (loss.view(shift_labels.size()) * shift_mask).sum() / shift_mask.sum()  
        return loss  

    def train_step(  
        self,  
        batch_texts: List[str],  
        pretrain_texts: List[str]  
    ) -> Dict[str, float]:  
        """执行一个训练步骤"""  
        self.model.train()  
        
        # 编码输入文本  
        inputs = self.tokenizer(  
            batch_texts,  
            padding=True,  
            truncation=True,  
            max_length=self.max_length,  
            return_tensors="pt"  
        ).to(self.device)  
        
        # 编码预训练文本  
        pretrain_inputs = self.tokenizer(  
            pretrain_texts,  
            padding=True,  
            truncation=True,  
            max_length=self.max_length,  
            return_tensors="pt"  
        ).to(self.device)  

        # 计算当前策略的输出  
        outputs = self.model(**inputs)  
        logits = outputs.logits  

        # 计算参考模型的输出  
        with torch.no_grad():  
            ref_outputs = self.ref_model(**inputs)  
            ref_logits = ref_outputs.logits  

        # 计算奖励  
        rewards = self.compute_rewards(batch_texts)  

        # 计算KL散度  
        kl_div = self.compute_kl_divergence(  
            logits,  
            ref_logits,  
            inputs['attention_mask']  
        )  

        # 计算预训练损失  
        pretrain_loss = self.pretrain_loss(  
            outputs.logits,  
            pretrain_inputs['input_ids'],  
            pretrain_inputs['attention_mask']  
        )  

        # 计算总损失  
        # PPO目标函数: reward - beta * KL + gamma * pretrain_loss  
        total_loss = -rewards.mean() + self.beta * kl_div + self.gamma * pretrain_loss  

        # 优化步骤  
        self.optimizer.zero_grad()  
        total_loss.backward()  
        self.optimizer.step()  

        return {  
            'total_loss': total_loss.item(),  
            'reward': rewards.mean().item(),  
            'kl_div': kl_div.item(),  
            'pretrain_loss': pretrain_loss.item()  
        }  

    def train(  
        self,  
        train_data: List[str],  
        pretrain_data: List[str],  
        num_epochs: int = 3  
    ):  
        """训练循环"""  
        wandb.init(project="gpt2-ppo-training")  
        
        for epoch in range(num_epochs):  
            total_stats = {  
                'total_loss': 0.,  
                'reward': 0.,  
                'kl_div': 0.,  
                'pretrain_loss': 0.  
            }  
            
            # 将数据分成批次  
            num_batches = len(train_data) // self.batch_size  
            for i in range(num_batches):  
                batch_texts = train_data[i*self.batch_size:(i+1)*self.batch_size]  
                pretrain_batch = np.random.choice(  
                    pretrain_data,  
                    size=self.batch_size  
                ).tolist()  
                
                # 执行训练步骤  
                stats = self.train_step(batch_texts, pretrain_batch)  
                
                # 更新统计信息  
                for k, v in stats.items():  
                    total_stats[k] += v  
                
                # 记录到wandb  
                wandb.log(stats)  
            
            # 计算epoch平均值  
            avg_stats = {  
                k: v / num_batches for k, v in total_stats.items()  
            }  
            print(f"Epoch {epoch+1}/{num_epochs}, Stats: {avg_stats}")  

    def generate(self, prompt: str, max_length: int = 100) -> str:  
        """使用训练后的模型生成文本"""  
        self.model.eval()  
        inputs = self.tokenizer(  
            prompt,  
            return_tensors="pt",  
            padding=True  
        ).to(self.device)  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                num_return_sequences=1,  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)  

# 使用示例  
def main():  
    # 准备训练数据（示例）  
    train_data = [  
        "这是一个训练样本。",  
        "这是另一个训练样本。",  
        # ... 更多训练数据  
    ]  
    
    pretrain_data = [  
        "这是预训练数据。",  
        "这是更多预训练数据。",  
        # ... 更多预训练数据  
    ]  
    
    # 初始化训练器  
    trainer = PPOTrainer(  
        model_name="gpt2",  
        lr=1e-5,  
        beta=0.2,  
        gamma=0.1,  
        batch_size=8  
    )  
    
    # 开始训练  
    trainer.train(train_data, pretrain_data, num_epochs=3)  
    
    # 生成示例  
    prompt = "请给我讲一个故事："  
    generated_text = trainer.generate(prompt)  
    print(f"Generated text: {generated_text}")  

if __name__ == "__main__":  
    main()