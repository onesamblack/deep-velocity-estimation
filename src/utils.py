from multiprocessing import Pool
import math
import pathlib
import glob
import cv2
import os
import numpy as np
from typing import Sequence, Union

class PreProcessingDefaults:
    crop_x = (0, -120)
    crop_y = (120, -160)
    sharpening_kernel_2ng_kernel_1 = np.array([[-1, -1, -1],[-1, 8, -1 ],[-1, -1, -1]])
    sharpening_kernel_2 = np.array([[0, -1, 0],[-1, 5, -1 ],[0, -1, 0]])
    sharpening_kernel_3 = np.array([[1/9, 1/9, 1/9], [1/9, 1, 1/9], [1/9, 1/9, 1/9]])


class OutputDefaults:
    resize_factor = 1.2
    codec = cv2.VideoWriter_fourcc(*'XVID')


class ProcessPipeline:
    """
    a processing pipeline is composed of multiple
    functions with the signature
        >>> f(x, ...)
    where x is an image (np.array)

    The pipeline applies the functions in `functions`
    linearly, with any arguments defined in `args`

    This is a convenience way of applying
    one or more transformations to frames
    of a video. It isn't bulletproof

    """
    def __init__(self, functions, args):
        self.functions = functions
        self.args = args

    def process_image(self, x):
        for i, f in enumerate(self.functions):
            if type(self.args[i]) == tuple:
                x = f(x, *self.args[i])
            if type(self.args[i]) == dict:
                x = f(x, **self.args[i])
            else:
                if self.args[i]:
                    x = f(x, self.args[i])
                else:
                    x = f(x)
        return x

def resize(x: np.ndarray, factor: float=2)-> np.ndarray:
    """
    resizes an  image by the factor. If factor
    < 1, reduces the size, >1 increases the size
    This doesn't perform interpolation

    Parameters
    -------------
    x: np.ndarray
        an image as an array
    factor: float
        the resize factor

    """
    return cv2.resize(x, (int(x.shape[0] * factor), 
                          int(x.shape[1] * factor)))

def append_text_to_img(img: np.ndarray, text: Sequence, colors: Sequence[tuple])-> np.ndarray:
    """
    Appends multiple lines of text to an image
    
    Parameters
    ----------
    img : np.ndarray
        the input image
    text : Sequence
        a list of text to append to the image. ["foo", "naz", "1028.23"]
        this will add a padding to each line of the text
    colors : Sequence
        a list of RGB color tuples. 
        TODO: default colors if none is provided
    
    Returns
    -------
    np.ndarray
        the image with the text
    """

    assert(len(text) == len(colors), "Must have a color for each line of text")
    size = cv2.getTextSize(text[0], 
                    OutputDefaults.text_font, 
                    OutputDefaults.text_size,  
                    OutputDefaults.text_thickness)
    output = None

    for i, txt_line in enumerate(text):
        if i == 0:
            pos = (5,(size[0][1] + OutputDefaults.text_padding))
        else:
            pos = (5, (\
                       ((size[0][1] * (i+1)) + OutputDefaults.text_padding) 
                       + (OutputDefaults.text_padding * i)))
        output = cv2.putText(img, 
                         txt_line, 
                         pos,  
                         OutputDefaults.text_font, 
                         OutputDefaults.text_size, 
                         colors[i], 
                         lineType =cv2.LINE_AA)

    return output

def colorize(x: np.ndarray) -> np.ndarray:
    """
    converts a gray image to a color image

    """
    return cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)

def output_video_from_frames(dir: str, 
                           output_filename: str,
                           codec,
                           fps: int,
                           size: tuple,
                           start=Union[None, int],
                           stop=Union[None, int]):
    """
    Creates an mp4 from a series of images in a
    directory. The images must use the following
    naming convention: 
        {i}.jpg , i != 0. Where i is the index of
        the frame in the output video

    
    Parameters
    ----------
    dir : str
        input directory
    output_filename : str
        name of output video
    codec : 
        cv2.codec
    fps : int
        frames per second in the output
    size : tuple
        size of input images
    start : None, optional
        if provided, starts building the video from that frame
    stop : None, optional
        if provided, stops building the video at that frame
    """
    out = cv2.VideoWriter(filename,
                          codec, 
                          fps, 
                          (size[0],size[1]))

    images = glob.glob(f"{dir}/*.jpg")
    images = sorted(images, key=lambda x: int(pathlib.Path(x).stem))

    print(f"{len(images)} in output video", flush=True)

    #iter over specified frames
    if start:
        stop = stop if stop is not None else len(images)
        for i in range(start,stop):
            x = cv2.imread(f"{dir}/{i}.jpg")
            out.write(x)
    # iter over all images
    else:
        for img in images:
            x = cv2.imread(img)
            out.write(x)

    out.release()


def agcwd(image: np.ndarray, w: float=0.5) -> np.ndarray:
    """
    Performs Adaptive Gamma Correction With Weighting Distribution 

    original source: https://github.com/qyou/AGCWD/blob/master/agcwd.py
    
    Parameters
    ----------
    image : np.ndarray
        input image
    w : float, optional
        gamma default: 0.5
    
    Returns
    -------
    np.ndarray
        adjusted image
    """
    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image


def extract_value_channel(color_image: np.ndarray) -> np.ndarray:
    """extracts value channels from a color image
    
    Parameters
    ----------
    color_image : np.ndarray
        input image
    
    Returns
    -------
    np.ndarray
        output image
    """
    color_image = color_image.astype(np.float32) / 255.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return np.uint8(v * 255)


def get_pdf(gray_image: np.ndarray) -> np.ndarray:
    """returns distribution over the image pixels
    
    Parameters
    ----------
    gray_image : np.ndarray
        input image
    
    Returns
    -------
    np.ndarray
        distribution
    """
    height, width = gray_image.shape
    pixel_count = height * width
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist / pixel_count


def set_value_channel(color_image: np.ndarray, value_channel:np.ndarray) -> np.ndarray:
    """
    Sets the value channel for a color image
    
    Parameters
    ----------
    color_image : np.ndarray
    value_channel : np.ndarray
        
    Returns
    -------
    np.ndarray
        The color image
    """
    value_channel = value_channel.astype(np.float32) / 255
    color_image = color_image.astype(np.float32) / 255.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image[:, :, 2] = value_channel
    color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
    return color_image

def gray(x: np.ndarray) -> np.ndarray:
    """grayscales an image
    
    Parameters
    ----------
    x : np.ndarray
        input image (color)
    
    Returns
    -------
    np.ndarray
        output image
    """
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

def crop(x: np.ndarray, crop_points: Sequence[tuple], color:bool=False) -> np.ndarray:
    """
    Crops an image using crop points
    
    Parameters
    ----------
    x : np.ndarray
        input image
    crop_points : Sequence[tuple]
        an iterable of crop points. iterable[0] should be the crop
        points in pixels for the x axis, iterable[1] should
        be the crop points in pixels for the y axis
    color : bool, optional
        True if the input is colored. default: False
    
    Returns
    -------
    np.ndarray
        The croppped image
    """
    def crop_index(axis, points):
        coordinates = [0, x.shape[axis]]
        for i, p in enumerate(points):
            if all([i == 0, p < 0]):
                raise Exception(f"can't use negative indices for the first position in axis:{axis}")
            else:
                if i == 0:
                    coordinates[0] = p
                else:
                    if p < 0:
                        # cuts off pixels defined
                        coordinates[i] = x.shape[axis] + p
                    elif p > 0:
                        coordinates[i] = p
                    else:
                        coordinates[i] = x.shape[axis]
        return coordinates
    
    crop_x = crop_index(0, crop_points[0])
    crop_y = crop_index(1, crop_points[1])
    if color:
        return x[ crop_x[0]:crop_x[1], \
             crop_y[0]: crop_y[1], :]
    else:
        return x[crop_x[0]:crop_x[1], \
             crop_y[0]: crop_y[1]]


# optical flow from https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



def farneback_optical_flow(x, x2, return_type='hsv'):

    prevgray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    prevgray = gray
    if return_type == "flow": 
        return draw_flow(gray, flow)
    if return_type == "hsv":
        return draw_hsv(flow)


def read_images_from_video(video: str, output_dir) -> None:
    """
    reads images from a video using ffmpeg.

    for each frame in the input video, this will output:
        1.jpg
        2.jpg
        ...
        for all frames in the video
    
    Parameters
    ----------
    video : str
        filename of the video
    output_dir : TYPE
        the directory to output images to
    """

    os.system(f"ffmpeg  -i \"{video}\" -q:v 1 \"{output_dir}/$filename%d.jpg\"")
  

def apply_kernel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolves a 2d kernel over the input image
    
    Parameters
    ----------
    x : np.ndarray
        input image
    kernel : np.ndarray
        the kernel to use
    
    Returns
    -------
    np.ndarray
        the convolved image
    """
    return cv2.filter2D(x, -1, kernel)
