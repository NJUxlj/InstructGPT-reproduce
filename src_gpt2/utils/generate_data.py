import json  
import random  

def generate_sft_data(num_samples=1000):  
    """生成SFT(监督微调)训练数据"""  
    prompts = [  
        "解释什么是人工智能：",  
        "如何写一篇好的论文？",  
        "请介绍一下Python编程语言：",  
        "什么是机器学习？",  
        "如何提高学习效率？",  
        "请解释区块链技术：",  
        "如何保持健康的生活方式？",  
        "请介绍一下量子计算：",  
        "如何有效管理时间？",  
        "解释什么是深度学习："  
    ]  
    
    responses = [  
        "人工智能是指通过程序实现类人的智能，包括学习、推理和自适应等能力。它是计算机科学的一个重要分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。",  
        "写好论文需要明确的研究问题、充分的文献综述、严谨的研究方法、清晰的论述结构和准确的数据分析。同时要注意格式规范，保持逻辑连贯性。",  
        "Python是一种高级编程语言，以简洁易读的语法著称。它支持多种编程范式，包括面向对象、命令式和函数式编程。Python具有丰富的标准库和第三方库，广泛应用于web开发、数据分析、人工智能等领域。",  
        "机器学习是人工智能的一个子领域，研究计算机如何通过经验自动改进算法性能。它通过数据训练模型，使计算机能够从经验中学习，而无需显式编程。",  
        "提高学习效率可以通过制定明确的学习计划、使用番茄工作法、保持专注、适时休息、及时复习等方法。同时保持良好的作息和饮食习惯也很重要。",  
        "区块链是一种分布式账本技术，通过密码学原理和共识机制确保数据的安全性和不可篡改性。它可以用于数字货币、智能合约、供应链管理等多个领域。",  
        "健康的生活方式包括规律作息、均衡饮食、适量运动、保持心理健康等方面。建议每天保证充足睡眠，多吃蔬菜水果，坚持运动，保持积极乐观的心态。",  
        "量子计算是利用量子力学原理进行计算的技术。它使用量子比特代替传统的二进制比特，可以同时处理多个状态，在某些特定问题上具有显著的性能优势。",  
        "有效的时间管理包括设定优先级、分解任务、避免拖延、善用工具等。建议使用时间管理矩阵，将任务按重要性和紧急性分类处理。",  
        "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑的学习过程。它能够自动学习特征，在图像识别、自然语言处理等领域取得了突破性进展。"  
    ]  
    
    data = []  
    for _ in range(num_samples):  
        idx = random.randint(0, len(prompts)-1)  
        data.append({  
            "prompt": prompts[idx],  
            "response": responses[idx]  
        })  
    
    return data  

def generate_reward_model_data(num_samples=500):  
    """生成奖励模型训练数据"""  
    prompts = [  
        "请写一首关于春天的诗：",  
        "如何解决环境污染问题？",  
        "请描述你理想中的未来：",  
        "如何看待人工智能发展？",  
        "请给出改善教育的建议："  
    ]  
    
    better_responses = [  
        "春风轻抚大地，唤醒沉睡的生命。繁花次第开放，鸟儿欢快鸣唱。新绿染遍山川，春意盎然处处。",  
        "环境污染问题需要多方面协同解决：1. 推广清洁能源使用 2. 加强环保法律监管 3. 提高公众环保意识 4. 发展循环经济 5. 加大环保技术研发投入。",  
        "在理想的未来中，科技与人文和谐共存，环境得到有效保护，教育资源公平分配，人类在探索宇宙奥秘的同时也不忘初心，守护地球家园。",  
        "人工智能发展应当以服务人类为本，在提高效率的同时注重伦理道德。我们既要积极拥抱新技术，也要防范潜在风险，确保AI发展的可控性和安全性。",  
        "改善教育建议：1. 推行个性化教学 2. 加强素质教育 3. 优化教育资源分配 4. 提升教师专业素养 5. 创新教学方法 6. 重视心理健康教育。"  
    ]  
    
    worse_responses = [  
        "春天到了，花开了，很漂亮。",  
        "环境污染很严重，政府应该管一管。",  
        "未来一定会更好的。",  
        "人工智能很厉害，但也有危险。",  
        "教育需要改革，现在的教育有问题。"  
    ]  
    
    data = []  
    for _ in range(num_samples):  
        idx = random.randint(0, len(prompts)-1)  
        data.append({  
            "prompt": prompts[idx],  
            "better_response": better_responses[idx],  
            "worse_response": worse_responses[idx]  
        })  
    
    return data  

def generate_pretrain_data(num_samples=2000):  
    """生成预训练数据"""  
    topics = [  
        "科技", "教育", "环境", "健康", "文化",  
        "经济", "社会", "艺术", "体育", "历史"  
    ]  
    
    texts = []  
    for topic in topics:  
        texts.extend([  
            f"关于{topic}的发展趋势分析...",  
            f"{topic}领域的最新研究进展...",  
            f"探讨{topic}面临的主要挑战...",  
            f"{topic}创新与未来展望...",  
            f"浅析{topic}的重要性..."  
        ])  
    
    data = []  
    for _ in range(num_samples):  
        text = random.choice(texts)  
        data.append({  
            "text": text + "这是一段详细的说明和分析内容。"  
        })  
    
    return data  

def main():  
    # 生成SFT数据  
    sft_data = generate_sft_data()  
    with open("data/sft_data.json", "w", encoding="utf-8") as f:  
        json.dump(sft_data, f, ensure_ascii=False, indent=2)  
    
    # 生成奖励模型数据  
    rm_data = generate_reward_model_data()  
    with open("data/rm_data.json", "w", encoding="utf-8") as f:  
        json.dump(rm_data, f, ensure_ascii=False, indent=2)  
    
    # 生成预训练数据  
    pretrain_data = generate_pretrain_data()  
    with open("data/pretrain_data.json", "w", encoding="utf-8") as f:  
        json.dump(pretrain_data, f, ensure_ascii=False, indent=2)  

if __name__ == "__main__":  
    # 创建data目录  
    import os  
    os.makedirs("data", exist_ok=True)  
    main()  
    print("数据生成完成！")