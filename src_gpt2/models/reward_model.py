import torch  
import torch.nn as nn  
from transformers import GPT2PreTrainedModel, GPT2Model  

class RewardModel(GPT2PreTrainedModel):  
    def __init__(self, config):  
        super().__init__(config)  
        self.gpt2 = GPT2Model(config)  
        self.value_head = nn.Linear(config.n_embd, 1)  
        self.init_weights()  

    def forward(  
        self,  
        input_ids=None,  
        attention_mask=None,  
        token_type_ids=None,  
        position_ids=None,  
        head_mask=None,  
        inputs_embeds=None,  
        return_dict=True,  
    ):  
        outputs = self.gpt2(  
            input_ids,  
            attention_mask=attention_mask,  
            token_type_ids=token_type_ids,  
            position_ids=position_ids,  
            head_mask=head_mask,  
            inputs_embeds=inputs_embeds,  
            return_dict=return_dict,  
        )  

        hidden_states = outputs[0]  
        rewards = self.value_head(hidden_states).squeeze(-1)  
        
        # 只使用序列最后一个token的reward作为整体reward  
        last_token_rewards = rewards * attention_mask  
        sequence_rewards = last_token_rewards.sum(dim=1) / attention_mask.sum(dim=1)  
        
        return sequence_rewards