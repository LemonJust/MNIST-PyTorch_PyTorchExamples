# MNIST - PyTorch

<p align="center">
<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/1000Images.PNG" width="600px" height = "300px" alt="MNIST digits"></img>
</p>

A repository linked in the following discussion : Errors when using num_workers>0 in DataLoader ( and low GPU usage on Windows ) https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4 

This repository contains code for training a LeNet-5 like Convnet on MNIST handwritten digits. 

## Requirements

1. torch
2. torchvision
3. Matplotlib
4. Numpy
5. pandas
6. tqdm
7. seaborn

You can download PyTorch and torchvision from their website. Download the library depending on your system hardware (whether you have CUDA enabled device or not). 

## Execution

In terminal type,
```bash
$ python3 MNIST.py
```

This will start training the model from scratch depending on the training configurations provided. The program will also output few graphs for your analysis.

## Training the Model

I have trained the model for 40 epochs in total. Training time was about 2 mins on my RTX 3080 GPU. However, training time depends on your hardware specifications.

The trained model is available [here](https://github.com/iVishalr/MNIST-PyTorch/blob/main/models/). This model achieves a test accuracy of 99.015% and a train accuracy of 99.56%. 

## Some eye pleasing stuffs

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture1.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture2.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture3.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture4.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture5.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

<img src="https://github.com/iVishalr/MNIST-PyTorch/blob/main/images/Capture6.PNG" width="600px" height = "300px" alt="MNIST digits"></img>

## License

MIT License
