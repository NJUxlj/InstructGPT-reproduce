import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from transformers import get_linear_schedule_with_warmup  
import wandb  
from tqdm import tqdm  
from typing import Dict, List  

class SFTTrainer:  
    def __init__(  
        self,  
        model,  
        tokenizer,  
        config: Dict,  
        device: torch.device  
    ):  
        """  
        初始化SFT训练器  
        
        Args:  
            model: GPT2模型  
            tokenizer: GPT2分词器  
            config: 训练配置  
            device: 训练设备  
        """  
        self.model = model.to(device)  
        self.tokenizer = tokenizer  
        self.config = config  
        self.device = device  
        
        # 优化器设置  
        self.optimizer = optim.AdamW(  
            self.model.parameters(),  
            lr=config["sft"]["learning_rate"],  
            weight_decay=0.01  
        )  
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)  
        
    def create_scheduler(self, num_training_steps: int):  
        """  
        创建学习率调度器  
        
        Args:  
            num_training_steps: 总训练步数  
        """  
        return get_linear_schedule_with_warmup(  
            self.optimizer,  
            num_warmup_steps=self.config["sft"]["warmup_steps"],  
            num_training_steps=num_training_steps  
        )  
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:  
        """  
        执行单个训练步骤  
        
        Args:  
            batch: 包含input_ids、attention_mask和labels的批次数据  
            
        Returns:  
            包含损失值的字典  
        """  
        input_ids = batch["input_ids"].to(self.device)  
        attention_mask = batch["attention_mask"].to(self.device)  
        labels = batch["labels"].to(self.device)  
        
        # 前向传播  
        outputs = self.model(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            labels=labels  
        )  
        
        loss = outputs.loss  
        
        # 反向传播  
        self.optimizer.zero_grad()  
        loss.backward()  
        
        # 梯度裁剪  
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  
        
        self.optimizer.step()  
        if hasattr(self, 'scheduler'):  
            self.scheduler.step()  
            
        return {  
            "loss": loss.item()  
        }  
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:  
        """  
        在验证集上评估模型  
        
        Args:  
            eval_dataloader: 验证数据加载器  
            
        Returns:  
            包含评估指标的字典  
        """  
        self.model.eval()  
        total_loss = 0  
        
        with torch.no_grad():  
            for batch in eval_dataloader:  
                input_ids = batch["input_ids"].to(self.device)  
                attention_mask = batch["attention_mask"].to(self.device)  
                labels = batch["labels"].to(self.device)  
                
                outputs = self.model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask,  
                    labels=labels  
                )  
                
                total_loss += outputs.loss.item()  
                
        avg_loss = total_loss / len(eval_dataloader)  
        return {"eval_loss": avg_loss}  
    
    def train(  
        self,  
        train_dataloader: DataLoader,  
        eval_dataloader: DataLoader = None,  
        save_path: str = None  
    ):  
        """  
        执行完整的训练过程  
        
        Args:  
            train_dataloader: 训练数据加载器  
            eval_dataloader: 验证数据加载器（可选）  
            save_path: 模型保存路径（可选）  
        """  
        print("Starting SFT training...")  
        
        # 初始化wandb  
        wandb.init(  
            project="instructgpt-sft",  
            config=self.config["sft"]  
        )  
        
        # 创建学习率调度器  
        num_training_steps = len(train_dataloader) * self.config["sft"]["num_epochs"]  
        self.scheduler = self.create_scheduler(num_training_steps)  
        
        # 训练循环  
        best_eval_loss = float('inf')  
        
        for epoch in range(self.config["sft"]["num_epochs"]):  
            self.model.train()  
            total_loss = 0  
            
            # 训练阶段  
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")  
            for step, batch in enumerate(progress_bar):  
                stats = self.train_step(batch)  
                total_loss += stats["loss"]  
                
                # 更新进度条  
                progress_bar.set_postfix(  
                    loss=stats["loss"],  
                    lr=self.scheduler.get_last_lr()[0]  
                )  
                
                # 记录到wandb  
                wandb.log({  
                    "train_loss": stats["loss"],  
                    "learning_rate": self.scheduler.get_last_lr()[0]  
                })  
            
            avg_train_loss = total_loss / len(train_dataloader)  
            print(f"Epoch {epoch+1}: Average train loss = {avg_train_loss:.4f}")  
            
            # 评估阶段  
            if eval_dataloader is not None:  
                eval_stats = self.evaluate(eval_dataloader)  
                print(f"Epoch {epoch+1}: Eval loss = {eval_stats['eval_loss']:.4f}")  
                
                wandb.log({  
                    "eval_loss": eval_stats['eval_loss'],  
                    "epoch": epoch + 1  
                })  
                
                # 保存最佳模型  
                if save_path and eval_stats['eval_loss'] < best_eval_loss:  
                    best_eval_loss = eval_stats['eval_loss']  
                    self.save_model(save_path)  
                    print(f"New best model saved with eval loss: {best_eval_loss:.4f}")  
        
        wandb.finish()  
        print("Training completed!")  
    
    def save_model(self, path: str):  
        """  
        保存模型和tokenizer  
        
        Args:  
            path: 保存路径  
        """  
        self.model.save_pretrained(path)  
        self.tokenizer.save_pretrained(path)  
    
    def generate(  
        self,  
        prompt: str,  
        max_length: int = 100,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        num_return_sequences: int = 1  
    ) -> List[str]:  
        """  
        使用训练后的模型生成文本  
        
        Args:  
            prompt: 输入提示  
            max_length: 最大生成长度  
            temperature: 采样温度  
            top_p: 核采样参数  
            num_return_sequences: 生成序列数量  
            
        Returns:  
            生成的文本列表  
        """  
        self.model.eval()  
        
        inputs = self.tokenizer(  
            prompt,  
            return_tensors="pt",  
            padding=True,  
            truncation=True,  
            max_length=self.config["model"]["max_length"]  
        ).to(self.device)  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                temperature=temperature,  
                top_p=top_p,  
                num_return_sequences=num_return_sequences,  
                pad_token_id=self.tokenizer.pad_token_id,  
                eos_token_id=self.tokenizer.eos_token_id,  
                do_sample=True  
            )  
        
        return [  
            self.tokenizer.decode(output, skip_special_tokens=True)  
            for output in outputs  
        ]