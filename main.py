
import time
from multiprocessing import Process, Manager

import torch

import numpy as np

from cpu import CPUInfer
from gpu import GPUInfer

from models import OR_Perceptron, XOR_Perceptron

class Main:
    def __init__(self):
        self.manager = Manager()
        self.return_dict = self.manager.dict()
        self.initial_time = time.time()

    def CPUInfer_module(self, data, model_list):
        obj_cpu = CPUInfer()
        t1 = time.time()
        output_res = obj_cpu.inference(data, model_list)
        t2 = time.time()
        self.return_dict['CPU_output'] = output_res
        print(f"Total CPU time : {t2 - t1}")

    def GPUInfer_module(self, data, model_list):
        obj_gpu = GPUInfer()
        t1 = time.time()
        output_res = obj_gpu.inference(data, model_list)
        t2 = time.time()
        self.return_dict['GPU_output'] = output_res
        print(f"Total GPU time : {t2 - t1}")

    def run(self, data, cpu_model_list, gpu_model_list):
        # running CPU models
        process1 = Process(target=self.CPUInfer_module, args=(data, cpu_model_list))
        process1.start()

        # running CPU models
        process2 = Process(target=self.GPUInfer_module, args=(data, gpu_model_list))
        process2.start()

        # joining the process
        process1.join()
        process2.join()
        
        print(f"Total time : {time.time() - self.initial_time}")
        
        return self.return_dict
    
if __name__ == "__main__":
    mainth = Main()

    or_model = XOR_Perceptron()
    or_model_cu = XOR_Perceptron(device='gpu')

    cpu_model_list = [or_model]
    gpu_model_list = [or_model_cu]

    data = [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ]

    for item in data:
        pred = mainth.run(item, cpu_model_list, gpu_model_list)

    # print("or(1, 1) = {}".format(perceptron_or(x=[1, 1], w=w_or[1:3], b=w_or[0]))) # 1
    # print("or(1, 0) = {}".format(perceptron_or(x=[1, 0], w=w_or[1:3], b=w_or[0]))) # 1
    # print("or(0, 1) = {}".format(perceptron_or(x=[0, 1], w=w_or[1:3], b=w_or[0]))) # 1
    # print("or(0, 0) = {}".format(perceptron_or(x=[0, 0], w=w_or[1:3], b=w_or[0]))) # 0

    
