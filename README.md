# Semantic Segmentation
### Introduction
In this project, I label the pixels of a road in images using a Fully Convolutional Network (FCN). You find the inference images of the most recent / best run in the `runs` folder.


### How this is approached
The network uses VGG-16 weights. The semantic segmentation network recreates the FCN-8s network ([Paper by Jonathan Long et al](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf))
The KITTI dataset is used to train and test the network.
- By using tensorflow the model and the weights are loaded for the encoder part
- The FCN model uses 1x1 convolutionals which makes the scaling for the decoder easier
- the decoder upsamples the input until we reach the original image size
- to get better results, skip layers are implemented
- the optimization is done by computing the softmax cross entropy between logits and labels.
- The cross entropy loss will be minimized with AdamOptimizer
- I got best results with 50 epochs, batch size of 5 and a learning rate of 0.001. The batch size is quite low. Reasons are, that the limited GPU memory does not allow a too large batch size. Further, the trainingsset is not to big - the batch size corresponds with it.


### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
Note: 2 GB of GPU memory is not enough to train and use this network.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
If you are using Anaconda, please find the gpu-environment configuration file in the project.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
