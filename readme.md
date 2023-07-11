
## Deep Learning Models for Emotion AI

I used deep learning to conduct emotion recognition on image data, by using the Facial Emotion Recognition (FER 2013) dataset. FER-2013 is one of the most popular openly-available emotion datasets, where images are labeled as one of 7 emotions. I performed transfer learning by using two different CNN architectures: ResNet50 and VGG16. Both architectures have been pretrained on Image Net, which is a visual database of more than 14 million hand-annotated images. I will fine-tune the models to FER2013 to compare performance and derive useful insights from training both networks.


## About Dataset 

### **Loading and Preparing the Data**

FER-2013 is a dataset containing 35,887 close-up images of human faces available. The photos are 48x48 grayscale images. FER2013 is one of the largest emotional expressions dataset openly available on Kaggle; it is well-studied but challenging to train on. Human-level accuracy has only been approximately 65%, and the best CNN architectures have reached approximately 75% accuracy. The classes are: [Angry, Happy, Sad, Fearful, Surprised, Disgusted, Neutral].

This cateogiration assumes Paul Ekman's emotion theory. As one of the most famous emotion psychologists, Paul Ekman proposed the idea that there are 6-8 basic, universal emotions that people feel no matter where they are from, and these emotions can be inferred from their facial expressions. There have been strong criticisms against this theory, for it's simplicity and shaky empirical evidence, but Paul Ekman is one of the cited emotion psychologists in computer vision because his theory fits the machine learning (classification) framework most conveniently of all the emotion theories so far. Other emotion theories, such as those that support multi-modal input for inferring emotions pose serious constraints on computational processing as well as finding good quality and widely-available data for different demographics. 

Regardless, in this assignment, FER2013 will be utilized. In order to do so, I perform the following processing steps to prepare the data for training. 

**1. Imitating RGB** 

FER2013 photos are grayscale, whereas both ResNet50 and VGG16 require a 3-channel RGB input to the neural network architecture. In order to emulate this, we can layer the grayscale with two additional channels that are duplicates of the original grayscale channel. Since the image will be the same overall three channels, the performance of the neural network should be the same as it was on RGB images.


**2. Class Imbalance**

FER2013 is a highly imbalanced dataset.  In FER-2013, the highest number of samples belong to the class “Happiness” with 7000 images, and the smallest class is the “Disgust” class with only a couple of hundred training samples. Most other classes are between 5000 and 3000 training samples. The State of The Art for VGG16 and ResNet50 both included using balanced datasets to improve misclassification rates for certain classes. In my implementation, I trained with both balanced and imbalanced data to understand how performance differs. 

Class imbalance is an important issue to address because it can render the accuracy metric an unreliable measure of classification performance. This is because the model can learn to predict the majority class for all examples and still end up with a high accuracy, which will deprioritize the model’s tendency to accurately predict minority classes. In order to deal with this issue, I chose the method of data resampling to balance out the class distribution. This includes undersampling the majority class of “Happiness”, and oversampling the minority class of “Disgust”.


## **ResNet50** 

### Compiling ResNet50

I loaded the pre-trained ResNet50 model that was trained on the Image Net dataset, and then froze the first 165 layers. I then added some additional dense layers to increase the depth of the model, with 'he Uniform' kernel initialization, which samples from a uniform distribution between $ (-\sqrt(6 / x), \sqrt(6 / x)) $ where x are the number of inputs in the weight tensor. Moreover, I added batch normalization layers. Batch Normalization is a effective in mini-batch gradient descent because it reparameterizes the distributions for each layer with a standardized mean of 0 and a standard deviation of 1 (Gaussian). This takes care of the problem of "internal covariate shift", which basically means that the model gets adjusted to different distributions in each layer and gets destabilized. By taking care of this issue, the learning algorithm is stabilized and can be expected to learn faster. Moreover, I added some dropout layers with 50% dropout. This reduced the number of trainable parameters from 15 million to 4 million, by randomly dropping nodes from some layers.

For the loss function, I used the sparse categorical cross-entropy loss, since my class labels are integers instead of one-hot encoded categorical representations. 


### Training ResNet50

After compiling the model, I ran it on the training data with a variable learning rate and an early stopping option. If the validation loss did not improve after a few epochs, the learning rate would be decayed by 10% to introduce sensitivity into the learning process. This would ensure that the model is not missing the local minimum. Moreover, the training would stop early using the Early Stopping function in Keras if no improvement happens over 15 epochs. I observed relatively overfitting even after adding 50% dropout to the FC layers and rebalancing the dataset. This could be improved by adding more data augmentations. 

### Testing ResNet50

My model achieved an accuracy of 53.20%, which is quite lower than the benchmark of 72.4% for FER2013 with ResNet50. This might be due to the fact that they were able to train it for much longer time and has used image generation augmentation during training, which was time-consuming in my implementation because the CPU and the GPU have to work in parallel while interleaving data augmentation with training, instead of doing it just once. 

One of the advantages of using ResNet50 is that it is a very deep model, with hundreds of sequential layers. However, unlike other very deep neural networks such as VGG, ResNet50 does not have the problem of vanishing gradients, which is to be expected in deep models where after backpropagating through the deep architecture, the gradients become zero prematurely and leave the initial layers unchanged. This is not a problem in ResNet50 because it has skip connections, which the gradients can flow directly from later layers to initial layers.

Since ResNet50 was trained on imagenet, which is a large database of images in the wild, FER2013 is just a collection of facial expressions, which means that instead of using weights from pretraining on general image data, if we could access a model that was also pre-trained on image data, then it could potentially increase the ability of the model to learn the nuances between facial expressions of people. Currently, the model knows how to tell lots of different kinds of things apart in image data, but is not necessarily concerned with telling faces apart that well. Perhaps, a face recognition dataset is more appropriate because once a model is well-acquainted with telling faces apart, we could use transfer learning to train it in telling variations in facial expressions apart. The SOTA implementation was directly done on VGGFace instead of VGG16, where the weights were more attuned to faces. This may explain the difference in performance. 

In transfer learning, one of the things that improved training was altering the batch sizes. Decreased batch sizes yielded better results on the validation set, which could be because smaller batches are more noisy, so they can help the model generalize better by not attuning to the noise or getting stuck in local minima. Smaller batch sizes are also better with using the SGD optimizer.



## **VGG16**

VGG was proposed in 2014 by Karen Simonyan and Andrew Zisserman from the Visual Geometry Group in Oxford University, hence the name VGG. The unique thing about VGG that made it a success at the time was that it used very small convolutional filters of 3x3, which allowed it to push the depth of the neural network to 16-19 layers without running into limitations the previous models ran into such as too many parameters. 

### Compiling VGG16

I froze all layers in VGG16 and added multiple fully-connected dense layers at the end with relu activations, as well as a softmax layer. The choice of nodes in the FC layers is consistent with the SOTA implementation.

### Training VGG16

At first, there was serious overfitting during training, with training accuracy at 90% contrasted with validation and test accuracy plateaued in the 40-45% range. In order to counter this, I added 30% dropout. This significantly decreased overfitting during training, which may explain the large variance in FER2013. To combat overfitting further, I performed training on augmented data using sklearn's Image Generator. At first, this led to very slow training because the data augmentation would be performed on the CPU whereas the training would happen on GPU. In order to deal with this issue, I turned multi-processing on and increased the worker size to 10, which allowed the augmentation and training to take place in parallel. Moreover, I increased batch size from 32 to 128, which significantly decreased training time from hours to minutes per epoch. This further decreased overfitting to the point that there was little difference between training and validation accuracy. Moreover, it slightly improved the model performance from 42% to 46%.  However, despite decrease in overfitting, model performance did not improve significantly. 

### Optimization

I used the Stochastic Gradient Descent optimizer with momentum. SGD differs from vanilla gradient descent because instead of running over all samples to update a paramater in an iteration, it only uses one or a subset of training samples. When using a subset, it is called mini-batch gradient descent. This makes it much faster than normal batch gradient descent, because SGD needs a much smaller set of training sample(s) to start improving. As such, a whole dataset does not need to be held in the RAM and vectorization can be more efficient.   

The stochastic in SGD refers to randomness, whereby random samples from the training data are chosen to update parameter during optimization. This means that instead of smooth learning curve, it will be much noisier. The benefit of this is that it can snap or jerk the learning out of local minima and increase the chances of it finding the global minimum. This is more appropriate for FER2013, because it is a challenging dataset to train on and does not have a smooth error manifold that is convex. The randomness, as such, is good for helping the model move out of local basins of attractions. 

The momentum in SGD can help take care of the instability introduced by the randomness, by accelerating the gradient descent in the relevant direction i.e. speed up movement in directions of strong improvement. The Nesterov momentum is a variation on momentum where if the gradient and momentum are pointing in different directions, is nesterov is set to True, then it redirects momentum in the right direction. 


### Testing VGG16

The overall performance of VGG16 plateaued when it reached 40-50 percent. Despite decreasing overfitting and following the steps applied in the SOTA implementation, performance did not improve significantly. 


## Comparing Architectures

When VGG16 was released, it was one of the very deep neural networks with upto 19 layers, until ResNet50 was introduced with hundreds of layers. This is possible through skip connections, which help it to fast track backpropagation. In my implementation, ResNet50 performed slightly better than VGG16, but not too much. Unlike the SOTA implementation, I used ResNet50 and VGG16 trained on Image Net weights, whereas the ones used by the winners of the FER2013 challenge used VGGFace weights. The other difference was that ResNet50 was more overfitted to the training set and it was harder to reduce overfitting in ResNet50 than it was in VGG16.


## **References**

‌Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements. Psychological Science in the Public Interest, 20(1), 1–68. https://doi.org/10.1177/1529100619832930

Brownlee, J. (2019, January 15). A Gentle Introduction to Batch Normalization for Deep Neural Networks. Machine Learning Mastery. https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/

Heaven, D. (2020). Why faces don’t always tell the truth about feelings. Nature, 578(7796), 502–504. https://doi.org/10.1038/d41586-020-00507-5

‌Khaireddin, Y., & Chen, Z. (n.d.). Facial Emotion Recognition: State of the Art Performance on FER2013. Retrieved March 29, 2022, from https://arxiv.org/pdf/2105.03588.pdf

Pramerdorfer, C., & Kampel, M. (2013). Facial Expression Recognition using Convolutional Neural Networks: State of the Art. Paperswithcode.com. https://paperswithcode.com/paper/facial-expression-recognition-using

Ruiz, P. (2018, October 8). Understanding and visualizing ResNets - Towards Data Science. Medium; Towards Data Science. https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

‌

‌
