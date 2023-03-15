
import time
from multiprocessing import Process, Manager

import torch

import numpy as np

from cpu import CPUInfer
from gpu import GPUInfer

from models import AND_Perceptron, OR_Perceptron, XOR_Perceptron


class Main:
    def __init__(self):
        self.manager = Manager()
        self.return_dict = self.manager.dict()
        self.initial_time = time.time()

    def cpuinfer_module(self, data, model_list):
        obj_cpu = CPUInfer()
        t1 = time.time()
        output_res = obj_cpu.inference(data, model_list)
        t2 = time.time()
        self.return_dict['CPU_output'] = output_res
        print(f"Total CPU time : {t2 - t1}")

    def gpuinfer_module(self, data, model_list):
        obj_gpu = GPUInfer()
        t1 = time.time()
        output_res = obj_gpu.inference(data, model_list)
        t2 = time.time()
        self.return_dict['GPU_output'] = output_res
        print(f"Total GPU time : {t2 - t1}")

    def run(self, data, cpu_model_list, gpu_model_list):
        # running CPU models
        process1 = Process(target=self.cpuinfer_module,
                           args=(data, cpu_model_list))
        process1.start()

        # running CPU models
        process2 = Process(target=self.gpuinfer_module,
                           args=(data, gpu_model_list))
        process2.start()

        # joining the process
        process1.join()
        process2.join()

        print(f"Total time : {time.time() - self.initial_time}")

        return self.return_dict


if __name__ == "__main__":
    mainth = Main()

    or_model = OR_Perceptron()
    and_model = AND_Perceptron()
    xor_model = XOR_Perceptron()

    or_model_cu = OR_Perceptron()
    and_model_cu = AND_Perceptron()
    xor_model_cu = XOR_Perceptron()

    cpu_model_list = [or_model, and_model, xor_model]
    gpu_model_list = [or_model_cu, and_model_cu, xor_model_cu]

    data = [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ]

    for item in data:
        pred = mainth.run(item, cpu_model_list, gpu_model_list)
        print(pred)
