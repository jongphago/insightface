import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from recognition.arcface_torch.backbones import get_model
from recognition.arcface_torch.configs.aihub_r50_onegpu import config as cfg
from datasets.AIHubDataset import AIHubDataset
from validate_aihub import validate_aihub

aihub_dataroot = "/home/jupyter/data/face-image/valid_aihub_family"
num_workers = 4
lfw_batch_size = 200
image_size = 112
aihub_mean = [0.5444, 0.4335, 0.3800]
aihub_std = [0.2672, 0.2295, 0.2156]

aihub_transforms = transforms.Compose(
    [
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=aihub_mean, std=aihub_std),
    ]
)

aihub_dataloader = torch.utils.data.DataLoader(
    dataset=AIHubDataset(
        dir=aihub_dataroot,
        pairs_path="data/pairs/valid/pairs_Family.txt",
        transform=aihub_transforms,
    ),
    batch_size=lfw_batch_size,
    num_workers=num_workers,
    shuffle=False,
)
