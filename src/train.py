import os
import gc
import uuid
import torch
import cv2
import glob
import sys
import datetime
import numpy as np
import collections
import pathlib
import warnings
import wandb
from tqdm import tqdm
from typing import Sequence, Union 
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn

from utils import *
from data import *
from model import DepthVelocityEstimationNet

USE_WANDB = False
RAW_IMAGES_DIRECTORY = "./data/train_raw"
PROCESSED_IMAGES_DIRECTORY = "./data/train"
DATA_ROOT_DIR = "./data"
RUN_TRAINING = False

if USE_WANDB: 
	import wandb
	from wandb.sdk import login


warnings.filterwarnings("ignore")

 
#create output directories
os.system(f"mkdir \"{PROCESSED_IMAGES_DIRECTORY}\"")
os.system(f"mkdir \"{RAW_IMAGES_DIRECTORY}\"")

# download data
if not os.path.exists(f"{DATA_ROOT_DIR}/train.mp4"):
	download_data(DATA_ROOT_DIR)


# reads all the images from the video into RAW_IMAGES_DIRECTORY/
read_images_from_video(f"{DATA_ROOT_DIR}/train.mp4", RAW_IMAGES_DIRECTORY)



# runs the preprocess pipeline on all the images into train/

preprocess_pipeline = ProcessPipeline([
                                       agcwd,
                                       apply_kernel, 
                                       crop, 
                                       resize], 
                            [
                      		{"w": 0.4},
                            {"kernel": PreProcessingDefaults.sharpening_kernel_2},
                            {'crop_points': (
                                PreProcessingDefaults.crop_x, 
                                PreProcessingDefaults.crop_y),
                             "color": True},
                            {"factor": 1.2}])



images = glob.glob(f"{RAW_IMAGES_DIRECTORY}/*.jpg")
images = sorted(images, key=lambda x: int(pathlib.Path(x).stem))

def preprocess_images(images):
    i = 0
    for f in images:
        s = pathlib.Path(f).stem
        try:
            x = cv2.imread(f"{RAW_IMAGES_DIRECTORY}/{s}.jpg")
            x = preprocess_pipeline.process_image(x)
            cv2.imwrite(f"{PROCESSED_IMAGES_DIRECTORY}/{s}.jpg", x)
            i += 1
            if i % 100 == 0:
                print(f"{i}", flush=True)   
        except:
            print(f"woops on {f}", flush=True)
            raise




train_labels = []
with open(f"{DATA_ROOT_DIR}/train.txt") as _f:
    _lines = _f.readlines()
    for v in _lines:
        train_labels.append(float(v))


train_labels = torch.tensor(train_labels, dtype=torch.float64)

                
# don't login until you've processed the images - wandb fails with multiprocess pools
if USE_WANDB:
    login(key=os.environ("WANDB_KEY"))

    wandb.init()

    config = wandb.config        
    config.batch_size = 32
    config.test_batch_size = 32    
    config.epochs = 100             
    config.lr = 1e-4             
    config.random_seed = 42             
    config.log_interval = 10
    config.validation_split = 0.2  
    config.frame_delta = 1
    config.depth = 20
    config.shuffle_dataset = True

else:
    Config = collections.namedtuple("ConfigDict", ["batch_size", 
    								 "test_batch_size", 
    								 "epochs", 
    								 "lr", 
    								 "random_seed", 
    								 "log_interval",
    								 "validation_split",
    								 "frame_delta",
    								 "depth", "shuffle_dataset"])


    config = Config(batch_size=32, 
    				test_batch_size=32, 
    				epochs=100, 
    				lr=1e-4, 
    				random_seed=42, 
    				log_interval=10, 
    				validation_split=0.2, 
    				frame_delta=1, 
    				depth=20,
                    shuffle_dataset=True)



def save_model(dir: str, 
			   model: nn.Module, 
			   epoch: int, 
			   name: str):

    """
    Saves model parameters
    
    Parameters
    ----------
    dir : str
        directory. saves each set of parameters under model_name/parameters/epoch.pt
    model : nn.Module
        a pytorch module
    epoch : int
        epoch number
    name : str
        the name of the model - overwrite this if not using wandb
    """
    pathlib.Path(f"{dir}/{name}/parameters/").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{dir}/{name}/parameters/_{epoch}.pt")
    print(f"saved {name} for epoch: {epoch}", flush=True)


def train(epoch, device):
    print(f"running train epoch: {epoch}", flush=True)
    model.train()
    train_loss = 0
    pbar = tqdm(total=len(train_indices))
    for i, data in enumerate(train_loader):
        x1s, x2s, labels = data[0].to(device, dtype=torch.float), \
                            data[1].to(device, dtype=torch.float), \
                             data[2].to(device, dtype=torch.float)
        preds = model(x1s, x2s)
        
        loss = loss_fn(preds, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()

        pbar.update(config.batch_size)
    
    per_item_loss = train_loss / len(train_loader)

    print(f"train_loss_per_item: {per_item_loss}")
    print(f"train_loss: {train_loss}")
    if USE_WANDB:
	    wandb.log({"train_loss_per_item": per_item_loss})
	    wandb.log({"train_loss": train_loss})


def validate(epoch,device):

    print(f"running validation epoch: {epoch}", flush=True)
    test_loss = 0
    model.eval()
    pbar = tqdm(total=len(val_indices))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x1s, x2s, labels = data[0].to(device, dtype=torch.float),\
                                data[1].to(device, dtype=torch.float),\
                                data[2].to(device, dtype=torch.float)
            test_preds = model(x1s, x2s)
            test_loss += loss_fn(test_preds, labels).item()
            pbar.update(config.test_batch_size)
    per_item_loss = test_loss / len(test_loader)
    print(f"test_loss_per_item: {per_item_loss}")
    print(f"test_loss: {test_loss}")

    if USE_WANDB:
	    wandb.log({"test_loss_per_item": per_item_loss})
	    wandb.log({"test_loss": test_loss})
  

if USE_WANDB:
	wandb.watch(model, log="all")

if RUN_TRAINING:
    _DEVICE = torch.device(f"cuda:{0}")
    model.to(_DEVICE)
    device_number = 0


if __name__ == "__main__":
    with Pool(2) as pool:
        pool.map(preprocess_images, [images[:math.ceil(len(images)/2)], images[math.ceil(len(images)/2):]])
    
    if RUN_TRAINING:
        
        model= DepthVelocityEstimationNet(3, depth=config.depth)
        
        dataset = MultiFrameDepthVideo(directory=PROCESSED_IMAGES_DIRECTORY, frame_delta=config.frame_delta, 
                                  Y=train_labels, read_grayscale=False, depth=config.depth)

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(1,dataset_size - config.frame_delta - config.depth))
        split = int(np.floor(config.validation_split * dataset_size))
        if config.shuffle_dataset:
            np.random.seed(config.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=config.test_batch_size, sampler=valid_sampler)


        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(1, config.epochs+1):
            try:
                train(epoch, _DEVICE)
                validate(epoch, _DEVICE)
                save_model(f"{PARAMETERS_ROOT_DIR}/runs", model, epoch,\
                          name= wandb.run.name if USE_WANDB else str(uuid.uuid4()))
            except:
                raise



