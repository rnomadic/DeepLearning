## CNN

<img src="images/CNN.png">

### Objective
Objective is to develop an algorithm that can be used as mobile or web app. App will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

### Model Selection
I decided to use CNN because I thought that using a series of convolutional layers with increasing filter size would allow the network to pick up increasingly complex patterns in the distinctions between the dog breeds (which is difficult to do with the naked eye). I also reasoned that having a pooling layer between each convolutional layer would increase the training efficiency of our model. I did decrease the number of filters in each convolutional layer because my intuition is that simpler models seem to perform better than more complex ones.

### Instruction
1. Setup your environment (for MAC only)
conda env create -f dog-mac.yml 

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location *dogImages*.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location *lfw*. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset. Place it in the repo, at location *bottleneck_features*.

5. Donwload the [ResNet 50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResNet50Data.npz). Place it in the repo, at location *bottleneck_features*.


### Saved Model
I have uploaded some of the trained model under folder *saved_model*. This can be loaded and tested with different set of data.
