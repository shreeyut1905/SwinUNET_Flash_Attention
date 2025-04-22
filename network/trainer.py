import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import math
from swin_unet_v2 import SwinTransformerSys
from dataset import CustomDataset
import wandb
import torch.optim as optim
import torchvision.transforms as transforms
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # entity="my-awesome-team-name",
    # Set the wandb project where this run will be logged.
    project="SwinUNET",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.0002,
        "architecture": "swin_unet",
        "dataset": "basic",
        "epochs": 100,
    },
)


config = wandb.config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinTransformerSys(in_chans=4,num_classes=4).to(device)
criterion = nn.BCEWithLogitsLoss()
# criterion = diceloss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
transform = transforms.Compose(
        [
            # transforms.Resize((args.image_size, args.image_size)),
        # transforms.ScaleIntensityRanged(a_min=0,a_max = 4343.01 , b_min=0,b_max=1,clip=True),
            transforms.ToTensor(),
            transforms.Normalize(mean = [18.78,21.48,26.79,15.22],std=[9.28,10.99,14.04,78.42]),
            transforms.Resize((224,224)),
        ]
    )
dataset = CustomDataset("C:\\Users\\Shreeyut\\deep-learning-lab\\UNET-SWIN\\SwinUNET_Flash_Attention\\network\\STARCOP\\ang20190922t192642_r2048_c0_w512_h512-20250410T125838Z-001", transform)
    # elif args.dataset == 'generic':
    #     transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
    #     transform_train = transforms.Compose(transform_list)
    #     dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=False)
    ## Define PyTorch data generator
training_generator = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True)
wandb.watch(model)
for epoch in range(config.epochs):
    running_loss = 0.0
    for img,mask in training_generator:
        img = img.float().to(device)
        mask = mask.float().to(device)
        print(img.shape)
        print(mask.shape)
        optimizer.zero_grad()
        output = model(img)
        print(output.shape)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(running_loss)
    
        wandb.log({"loss": running_loss })
    wandb.log({"epoch": epoch})

wandb.finish()