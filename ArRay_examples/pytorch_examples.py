"""
Profile different models with different batch size,
and plot the results.
"""
from ray.util.actor_pool import ActorPool
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,resnet152,vgg19,vit_h_14,resnet50,swin_v2_b, swin_v2_s,mobilenet_v3_small
from torchvision.models import convnext_large,squeezenet1_1,vit_l_16,vit_b_16
from pytroch_inference_actor import Predictor
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

import time
import argparse
if __name__ == "__main__":
    model =resnet50()
    batch_sizes = [512]
    # batch_sizes = [1,2,4,8,16,32]
    num_actors = len(batch_sizes)

    times=[]
    gpu_utils=[]
    bsizes=[]

    for i in range(0,num_actors):
        actor = Predictor(model)
        start = time.time()
        bs, gpu_util = actor.predict("~/data_ArRay",batch_sizes[i])
        end =time.time()
        times.append(end-start)
        gpu_utils.append(gpu_util)
        bsizes.append(bs)
        print(f"Duration: {end-start}")
        print(f"GPU_Util: {gpu_util}")



"""
    # 构建数据
    x = [str(x) for x in bsizes]
    y = gpu_utils
    z = times
    
    
    # 绘柱状图
    plt.bar(x=x, height=z, width = 0.25, label='run_times', color='tomato', alpha=0.8)

    # 在左侧显示图例
    plt.legend(loc="upper left")
    
    # 设置标题
    plt.title("resnet50")
    # 为两条坐标轴设置名称
    plt.xlabel("Batch_size")
    plt.ylabel("Run_time")

    
    # 画折线图
    ax2 = plt.twinx()
    ax2.set_ylabel("gpu_util")
    # 设置坐标轴范围
    ax2.set_ylim([0, 1])
    plt.plot(x, y, color='royalblue', marker='o', ls='-.', linewidth='1', label="gpu_util")
    # 显示数字
    for a, b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
    # 在右侧显示图例
    plt.legend(loc="upper right")

    # 保存图片
    plt.savefig('./figure/resnet50-profile.svg', bbox_inches='tight')
    # 显示图片
    plt.show()
"""