import os  
import torch  
import yaml  
import argparse  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  
from torch.utils.data import DataLoader, random_split  
from models.reward_model import RewardModel  
from trainers.sft_trainer import SFTTrainer  
from trainers.reward_trainer import RewardTrainer  
from trainers.ppo_trainer import PPOTrainer  
from utils.data_utils import SFTDataset, RewardModelDataset  

def load_config(config_path: str = "config/config.yaml"):  
    """加载配置文件"""  
    with open(config_path, "r", encoding='utf-8') as f:  
        return yaml.safe_load(f)  

def setup_args():  
    """设置命令行参数"""  
    parser = argparse.ArgumentParser(description='InstructGPT Training')  
    parser.add_argument('--stage', type=str, default='all',  
                      choices=['sft', 'reward', 'ppo', 'all'],  
                      help='Training stage to run')  
    parser.add_argument('--config', type=str, default='config/config.yaml',  
                      help='Path to config file')  
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',  
                      help='Directory to save model checkpoints')  
    return parser.parse_args()  

def setup_environment(config):  
    """设置训练环境"""  
    # 设置设备  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device}")  
    
    # 创建checkpoint目录  
    os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)  
    
    return device  

def train_sft(config, device, tokenizer):  
    """执行SFT训练阶段"""  
    print("\n=== Starting SFT Training ===")  
    
    # 加载模型  
    model = GPT2LMHeadModel.from_pretrained(config["model"]["base_model"])  
    
    # 准备数据  
    dataset = SFTDataset(  
        data_path="data/sft_data.json",  
        tokenizer=tokenizer,  
        max_length=config["model"]["max_length"]  
    )  
    
    # 分割训练集和验证集  
    train_size = int(0.9 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  
    
    train_dataloader = DataLoader(  
        train_dataset,  
        batch_size=config["sft"]["batch_size"],  
        shuffle=True  
    )  
    
    val_dataloader = DataLoader(  
        val_dataset,  
        batch_size=config["sft"]["batch_size"],  
        shuffle=False  
    )  
    
    # 初始化训练器  
    trainer = SFTTrainer(  
        model=model,  
        tokenizer=tokenizer,  
        config=config,  
        device=device  
    )  
    
    # 开始训练  
    trainer.train(  
        train_dataloader=train_dataloader,  
        eval_dataloader=val_dataloader,  
        save_path=os.path.join(config["checkpoint_dir"], "sft_model")  
    )  
    
    return model  

def train_reward_model(config, device, tokenizer):  
    """执行奖励模型训练阶段"""  
    print("\n=== Starting Reward Model Training ===")  
    
    # 加载模型  
    model = RewardModel(GPT2LMHeadModel.from_pretrained(config["model"]["base_model"]).config)  
    
    # 准备数据  
    dataset = RewardModelDataset(  
        data_path="data/rm_data.json",  
        tokenizer=tokenizer,  
        max_length=config["model"]["max_length"]  
    )  
    
    # 分割训练集和验证集  
    train_size = int(0.9 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  
    
    train_dataloader = DataLoader(  
        train_dataset,  
        batch_size=config["reward_model"]["batch_size"],  
        shuffle=True  
    )  
    
    val_dataloader = DataLoader(  
        val_dataset,  
        batch_size=config["reward_model"]["batch_size"],  
        shuffle=False  
    )  
    
    # 初始化训练器  
    trainer = RewardTrainer(  
        model=model,  
        tokenizer=tokenizer,  
        config=config,  
        device=device  
    )  
    
    # 开始训练  
    trainer.train(  
        train_dataloader=train_dataloader,  
        eval_dataloader=val_dataloader,  
        save_path=os.path.join(config["checkpoint_dir"], "reward_model")  
    )  
    
    return model  

def train_ppo(config, device, tokenizer, sft_model, reward_model):  
    """执行PPO训练阶段"""  
    print("\n=== Starting PPO Training ===")  
    
    # 创建参考模型（从SFT模型复制）  
    ref_model = GPT2LMHeadModel.from_pretrained(  
        os.path.join(config["checkpoint_dir"], "sft_model")  
    )  
    
    # 准备数据  
    dataset = SFTDataset(  
        data_path="data/pretrain_data.json",  
        tokenizer=tokenizer,  
        max_length=config["model"]["max_length"]  
    )  
    
    train_dataloader = DataLoader(  
        dataset,  
        batch_size=config["ppo"]["batch_size"],  
        shuffle=True  
    )  
    
    # 初始化训练器  
    trainer = PPOTrainer(  
        model=sft_model,  
        ref_model=ref_model,  
        reward_model=reward_model,  
        tokenizer=tokenizer,  
        config=config,  
        device=device  
    )  
    
    # 开始训练  
    trainer.train(  
        train_dataloader=train_dataloader,  
        save_path=os.path.join(config["checkpoint_dir"], "ppo_model")  
    )  
    
    return sft_model  

def main():  
    """主函数"""  
    # 解析命令行参数  
    args = setup_args()  
    
    # 加载配置  
    config = load_config(args.config)  
    config["checkpoint_dir"] = args.checkpoint_dir  
    
    # 设置环境  
    device = setup_environment(config)  
    
    # 加载tokenizer  
    tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["base_model"])  
    tokenizer.pad_token = tokenizer.eos_token  
    
    # 根据指定阶段执行训练  
    if args.stage in ['sft', 'all']:  
        sft_model = train_sft(config, device, tokenizer)  
    else:  
        sft_model = GPT2LMHeadModel.from_pretrained(  
            os.path.join(config["checkpoint_dir"], "sft_model")  
        )  
    
    if args.stage in ['reward', 'all']:  
        reward_model = train_reward_model(config, device, tokenizer)  
    else:  
        reward_model = RewardModel.from_pretrained(  
            os.path.join(config["checkpoint_dir"], "reward_model")  
        )  
    
    if args.stage in ['ppo', 'all']:  
        final_model = train_ppo(config, device, tokenizer, sft_model, reward_model)  
        
        # 保存最终模型  
        final_model.save_pretrained(  
            os.path.join(config["checkpoint_dir"], "final_model")  
        )  
        tokenizer.save_pretrained(  
            os.path.join(config["checkpoint_dir"], "final_model")  
        )  
    
    print("\n=== Training Complete ===")  
    
    # 测试生成  
    test_prompt = "请给我讲一个关于人工智能的故事："  
    print(f"\nTesting generation with prompt: {test_prompt}")  
    
    final_model = GPT2LMHeadModel.from_pretrained(  
        os.path.join(config["checkpoint_dir"], "final_model")  
    ).to(device)  
    
    inputs = tokenizer(  
        test_prompt,  
        return_tensors="pt",  
        padding=True,  
        truncation=True,  
        max_length=config["model"]["max_length"]  
    ).to(device)  
    
    with torch.no_grad():  
        outputs = final_model.generate(  
            **inputs,  
            max_length=200,  
            temperature=0.7,  
            top_p=0.9,  
            num_return_sequences=1,  
            pad_token_id=tokenizer.pad_token_id,  
            eos_token_id=tokenizer.eos_token_id,  
            do_sample=True  
        )  
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    print(f"\nGenerated text:\n{generated_text}")  

if __name__ == "__main__":  
    main()