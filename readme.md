# Deep Learning for Emotion Recognition 


I used deep learning to conduct emotion recognition on image data, by using the Facial Emotion Recognition (FER 2013) dataset. FER-2013 is one of the most popular openly-available emotion datasets, where images are labeled as one of 7 emotions. I performed transfer learning by using the ResNet50 neural network that was pre-trained on the Image Net dataset to conduct emotion recognition.

# Model Training -- ResNet50

I loaded the pre-trained ResNet50 model that was trained on the Image Net dataset, and then froze the first 143 layers. I then added some additional dense layers to increase the depth of the model, with 'he Uniform' kernel initialization, which samples from a uniform distribution between (-sqrt(6 / x), sqrt(6 / x)) where x are the number of inputs in the weight tensor. Moreover, I added batch normalization layers. Batch Normalization is a effective in mini-batch gradient descent because it reparameterizes the distributions for each layer with a standardized mean of 0 and a standard deviation of 1 (Gaussian). This takes care of the problem of "internal covariate shift", which basically means that the model gets adjusted to different distributions in each layer and gets destabilized. By taking care of this issue, the learning algorithm is stabilized and can be expected to learn faster. Moreover, I added some dropout layers with 50% dropout. This reduced the number of trainable parameters from 15 million to 4 million, by randomly dropping nodes from some layers. 

For the loss function, I used the sparse categorical cross-entropy loss, since my class labels are integers instead of one-hot encoded categorical representations. Adam is used as the optimizer since it generally performs well on ResNet50. 

After compiling the model, I ran it on the training data with a variable learning rate and an early stopping option. If the validation loss did not improve after a few epochs, the learning rate would be decayed by 10% to introduce sensitivity into the learning process. This would ensure that the model is not missing the local minimum. 

My model achieved an accuracy of 53.20%, which is lower than the benchmark of 72.4% for FER2013 with ResNet50.

One of the advantages of using ResNet50 is that it is a very deep model, with hundreds of sequential layers. However, unlike other _very_ deep neural networks such as VGG, ResNet50 does not have the problem of vanishing gradients, which is to be expected in deep models where after backpropagating through the deep architecture, the gradients become zero prematurely and leave the initial layers unchanged. This is not a problem in ResNet50 because it has skip connections, which the gradients can flow directly from later layers to initial layers. 

Since ResNet50 was trained on imagenet, which is a large database of images in the wild, FER2013 is just a collection of facial expressions, which means that instead of using weights from pretraining on general image data, if we could access a model that was also pre-trained on image data, then it could potentially increase the ability of the model to learn the nuances between facial expressions of people. Currently, the model knows how to tell lots of different kinds of things apart in image data, but is not necessarily concerned with telling faces apart that well. Perhaps, a face recognition dataset is more appropriate because once a model is well-acquainted with telling faces apart, we could use transfer learning to train it in telling variations in facial expressions apart. 

In transfer learning, one of the things that improved training was altering the batch sizes. Decreased batch sizes yielded better results on the validation set, which could be because smaller batches are more noisy, so they can help the model generalize better by not attuning to the noise.

# References
 

Ruiz, P. (2018, October 8). Understanding and visualizing ResNets - Towards Data Science. Medium; Towards Data Science. https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

‌Khaireddin, Y., & Chen, Z. (n.d.). Facial Emotion Recognition: State of the Art Performance on FER2013. Retrieved March 29, 2022, from https://arxiv.org/pdf/2105.03588.pdf

Brownlee, J. (2019, January 15). A Gentle Introduction to Batch Normalization for Deep Neural Networks. Machine Learning Mastery. https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/

‌