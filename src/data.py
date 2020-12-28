import os
import cv2
import glob
import torch
from torch.utils.data import (
	Dataset, 
	DataLoader
)


def download_data(directory) -> None:
    """
    downloads train.mp4 and train.txt
    for the challenge


    Parameters
    ----------
    directory: str
    	the output directory

    """

    os.system(f"curl https://github.com/commaai/speedchallenge/raw/master/data/train.mp4 -L -o {directory}/train.mp4")
    os.system(f"curl https://raw.githubusercontent.com/commaai/speedchallenge/master/data/train.txt -o {directory}/train.txt")



class MultiFrameVideo(Dataset):

    def __init__(self, Y, directory,
                 frame_delta= 1,
                 processing_pipeline=None, 
                 read_grayscale=False):
        self.Y = Y
        self.directory = directory
        self.frame_delta = frame_delta
        self.processing_pipeline = processing_pipeline
        self.read_grayscale = read_grayscale
    def __len__(self):
        return len(glob.glob(f"{self.directory}/*.jpg"))

    def _img(self, index):
        if self.read_grayscale:
            x = cv2.imread(f"{self.directory}/{index}.jpg", cv2.IMREAD_GRAYSCALE)  
        else:
            x = cv2.imread(f"{self.directory}/{index}.jpg")
        if self.processing_pipeline:
            x = self.processing_pipeline.process_image(x)
        return torch.tensor(x, dtype=torch.float64)


    def __getitem__(self, index):
        """
        Returns two frames, separated by `frame_deltas`
        """
        _x1 = self._img(index).item().T
        _x2 = self._img(index + self.frame_delta).item().T
        # return the mean of the two speeds from
        # each frame
        _y = torch.mean(self.Y[index:index+(self.frame_delta+1)])
        return _x1, _x2, _y


class OpticalFlowData(Dataset):
    def __init__(self, directory, Y):
        self.directory = directory
        self.Y = Y
        

    def __len__(self):
        return len(glob.glob(f"{self.directory}/*.jpg")) -1

    def _img(self, index):
        x = cv2.imread(f"{self.directory}/{index}.jpg")
        try:
            return torch.tensor(x, dtype=torch.float64)
        except TypeError as e:
            print(index)
            raise

    def __getitem__(self, index):
        """
        Returns two frames, separated by `frame_deltas`
        """
        img_index = index + 1
        _x1 = self._img(img_index).T
        _y = torch.mean(self.Y[index:index+1])
        return _x1, _y

class MultiFrameDepthVideo(MultiFrameVideo):

    def __init__(self, depth, **kwargs):
        super(MultiFrameDepthVideo, self).__init__(**kwargs)
        self.depth = depth

    def get_multiple_images(self, index):
        x1s = []
        x2s = []
        for i in range(index, index+self.depth):
            x1s.append(self._img(index).T)
            x2s.append(self._img(index + self.frame_delta).T)
        # concat along depth
        x1 = torch.stack([x for x in x1s], 0)
        x2 = torch.stack([x for x in x2s], 0)
    
        return x1, x2

    

    def __getitem__(self, index):
        """
        Returns two frames, separated by `frame_deltas`
        """
        if self.read_grayscale:
            _x1, _x2 = self.get_multiple_images(index)
            _x1 = _x1.unsqueeze(-1)
            _x2 = _x2.unsqueeze(-1)
        else:
            _x1, _x2 = self.get_multiple_images(index)

        # return the mean of the two speeds from
        # each frame
        _y = self.Y[index:index+(self.depth)].item()
        _y2 = self.Y[index+ self.frame_delta:index+(self.frame_delta + self.depth)].item()
        ys = torch.stack([_y, _y2],0)
        y = torch.mean(ys,0)
        return _x1, _x2, y