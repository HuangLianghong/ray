import torch
import ray
import os
import torchvision.transforms as transforms
from filelock import FileLock
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


@ray.remote(num_gpus=1)
class Predictor:
    def __init__(self, model):
        torch.cuda.is_available()
        self.model = model.to(torch.device("cuda"))
        self.model.eval()

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        
    def predict(self, data_dir):
        data_dir=os.path.expanduser(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        with FileLock(os.path.join(data_dir, ".ray.lock")):
            validation_dataset = CIFAR10(
                root=data_dir, train=True, download=True, transform=self.transform_test
            )
        validation_loader = DataLoader(validation_dataset, batch_size=128)   
        result=[]
        with torch.no_grad():
            for X, _ in validation_loader:
                X=X.cuda()
                result.append(self.model(X))


        # Write out the prediction result.
        # NOTE: unless the driver will have to further process the
        # result (other than simply writing out to storage system),
        # writing out at remote task is recommended, as it can avoid
        # congesting or overloading the driver.
        # ...

        # Here we just return the size about the result in this example.
        return len(result)