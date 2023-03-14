import os
import torch 

class GPUInfer:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    def inference(self, data, models):
        output_results = {}
        for model in models:
            output_results[model.name] = model.inference(data)
            
        return output_results