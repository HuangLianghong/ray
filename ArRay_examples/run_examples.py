from ray.util.actor_pool import ActorPool
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,resnet152,vgg19
from inference_actor import Predictor
from torch.utils.data import DataLoader, Subset
import ray
import time
import argparse
if __name__ == "__main__":
    ray.init(_temp_dir="/mnt/sda/2022-0526/home/hlh/ray-log-files/")
    model = vgg19()
    model_ref = ray.put(model)
    num_actors = 1
    actors = [Predictor.remote(model_ref) for _ in range(num_actors)]
    pool = ActorPool(actors)

    pool.submit(lambda a, v: a.predict.remote(v), "~/data_ArRay")
    while pool.has_next():
        print("Prediction output size:", pool.get_next())