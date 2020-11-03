from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('./data', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
