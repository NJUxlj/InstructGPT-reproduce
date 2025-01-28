from datasets import load_dataset
import re




# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    
    def __init__(self):
        pass
    
    
    def get_train_data(self):
        pass
    
    def get_eval_data(self):
        pass

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self):
        pass
    
    
    