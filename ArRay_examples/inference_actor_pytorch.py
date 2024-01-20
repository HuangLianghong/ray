import torch
import ray
import os
import time
import GPUtil
import torchvision.transforms as transforms
from filelock import FileLock
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from threading import Thread


class Predictor:
    def __init__(self, model):
        torch.cuda.is_available()
        self.model = model.to(torch.device("cuda"))
        self.model.eval()

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.Resize(224,antialias=True) # for  vit_h_14()

            ]
        )
    def test(self):
        print('Test whether this method can be call by the proxy_actor...')
        # raise Exception('Call the method in remote actor!')

    def profile(self):
        """
        This method is used to profile the GPU resources that this actor needs.
        """
        data_dir = "~/data_ArRay"
        data_dir=os.path.expanduser(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        with FileLock(os.path.join(data_dir, ".ray.lock")):
            validation_dataset = CIFAR10(
                root=data_dir, train=True, download=True, transform=self.transform_test
            )
        subset_size=2048
        subset = Subset(validation_dataset,range(subset_size))
        validation_loader = DataLoader(subset, batch_size=128)   
        result=[]
        with torch.no_grad():
            for X, _ in validation_loader:
                X=X.cuda()
                result.append(self.model(X))



        return
    
    def predict(self, data_dir, batch_size):
        data_dir=os.path.expanduser(data_dir)
        os.makedirs(data_dir, exist_ok=True)   
        with FileLock(os.path.join(data_dir, ".ray.lock")):
            validation_dataset = CIFAR10(
                root=data_dir, train=True, download=True, transform=self.transform_test
            )
        # validation_loader = DataLoader(validation_dataset, batch_size=batch_size) 

        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

        result=[]
        
        monitor = Monitor(0.05)
        with torch.no_grad():
            for X, _ in validation_loader:
                X=X.cuda()
                result.append(self.model(X))
        
        monitor.stop()
        # Write out the prediction result.
        # NOTE: unless the driver will have to further process the
        # result (other than simply writing out to storage system),
        # writing out at remote task is recommended, as it can avoid
        # congesting or overloading the driver.
        # ...

        # Here we just return the size about the result in this example.
        return batch_size, monitor.gpu_util[2]
    
class Monitor(Thread):

    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time-second between calls to GPUtil
        self.gpus=GPUtil.getGPUs()
        self.gpu_util=[0] * len(self.gpus)
        # self.count=0
        # self.memory_usage= [0] * len(self.gpus)
        self.start()
        
    def run(self):
        while not self.stopped:
            self.gpus=GPUtil.getGPUs()
            # self.count += 1
            for i in range(len(self.gpus)):
                if self.gpus[i].load>self.gpu_util[i]:
                    self.gpu_util[i] = self.gpus[i].load
                # if self.gpus[i].memoryUtil > self.memory_usage[i]:
                #     self.memory_usage[i] = self.gpus[i].memoryUtil
            time.sleep(self.delay)


    def stop(self):
        self.stopped = True
