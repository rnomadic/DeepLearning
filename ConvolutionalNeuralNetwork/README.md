## CNN

<img src="images/CNN.png">

### Objective
Objective is to develop an algorithm that can be used as mobile or web app. App will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

### Model Selection
I decided to use CNN because I thought that using a series of convolutional layers with increasing filter size would allow the network to pick up increasingly complex patterns in the distinctions between the dog breeds (which is difficult to do with the naked eye). I also reasoned that having a pooling layer between each convolutional layer would increase the training efficiency of our model. I did decrease the number of filters in each convolutional layer because my intuition is that simpler models seem to perform better than more complex ones.
