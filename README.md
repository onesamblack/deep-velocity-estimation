# Deep Velocity Estimation



## Background

Speed estimation from a single dashboard camera using Deep Convolutional Networks. This is a response to the challenge posed here: https://github.com/commaai/speedchallenge

There is an associated blog post [here](https://towardsdatascience.com/deep-convolutional-networks-for-monocular-velocity-estimation-a0081d6bc7a9)

See the post for more details on the experiments and the associated model architectures



## Results

| Experiment                                                   | MSE   |
| ------------------------------------------------------------ | ----- |
| Deep Velocity Estimation (Grayscale Input, Frame Delta: 1)   | > 10  |
| Deep Velocity Estimation (RGB Input, Frame Delta: 1)         | > 10  |
| Deep Convolutional Network with Farneback Flow (RGB Input)   | > 10  |
| DeepER Velocity Estimation (RGB Input, Depth: 20, Frame Delta: 1) | < 1** |

** Looking for someone to independently verify the performance, if you verify, please submit an issue with your results



## Usage

### Installation

Processing the images requires FFMPEG. See the installation guidelines [here](https://ffmpeg.org/download.html) for your platform

To install the python requirements, run:

```
pip install -r requirements.txt
```

### Run model training

After installing the requirements, from the root directory of the repo, run

```bash
python3 src/train.py
```


## Verification

As referenced above, the MSE on the model is less than 1, which could be close to SOTA. I'm looking for someone to verify the results I published on the blog.

If you do want to verify, please include the following details with your verification:
 
 - Platform
 - GPU Type
 - Batch Size
 - Any modifications to the parameters run

The results of the run are located [here](https://wandb.ai/sam-black/uncategorized/reports/Results-from-Deep-Velocity-Estimation--VmlldzozODM5ODY)



## References

See the [blog post]((https://towardsdatascience.com/deep-convolutional-networks-for-monocular-velocity-estimation-a0081d6bc7a9)) for references to relevant papers.







