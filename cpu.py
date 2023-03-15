import os

class CPUInfer:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    def inference(self, data, models):
        output_results = {}
        for model in models:
            output_results[model.name] = model.inference(data)
            
        return output_results