
# Intro to Neural Networks

# Background

Neural networks have been around for a while. They are over 70 years old, dating back to  their proposal in 1944 by Warren McCullough and Walter Pitts. These first proposed neural nets had thresholds and weights, but no layers and no specific training mechanisms.

The "Perceptron" the first trainable neural network was created by Frank Rosenblatt in 1957. It consisted of a single layer with adjustable weights in the middle of input and output layers.



![peceptron](img/nn-diagram.png)

## Inspiration from Actual Neurons

The composition of neural networks can bee **loosely** compared to a neuron.  We will be using it as an analogy to help us remember what is going on, but really, the comparison should stop there.

![neuron](img/neuron.png)

This is a loose analogy, but can be a helpful **mneumonic** (If I don't keep stressing this, my biologist sister might put out a hit on me). The inputs to our node are like inputs to our neurons.  They are either direct sensory information (our features) or input from other axons (nodes passing information to other nodes).  The body of our neuron (soma) is where the signals of the dentrites are summed together, which is loosely analogous to our **collector function**. If the summed signal is large enough (our **activation function**), they trigger an action potential which travels down the axon to be passed as output to other dentrites ([wikipedia neuron article](https://en.wikipedia.org/wiki/Neuron)). 

# Forward Propogation

Let's first look at **forward propogation** on the level of the perceptron.

We will use the built in dataset of handwritten numbers from sklearn, which comes from the UCI Machine Learning collection [digits source](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits). Each record is a 8 by 8 bit image of a handwritten number between 0 and 9. Each pixel value (a number between 0 and 16) represents the relative brightness of the pixel. 

It is similar to the famous [**MNIST**](http://yann.lecun.com/exdb/mnist/index.html) dataset which is sometimes referred to the "hello world" of computer vision [source](https://www.kaggle.com/c/digit-recognizer).  

With one input of pixels from a number, our input/output process looks like so:

When passing the data into our perceptron, we will flatten the image into a 64x1 array.

Our weights vector will have the same number of weights as pixels

![weights](img/log-reg-nn-ex-w.png)

We will instantiate our weight with small random numbers.


# Question: What shape should our weight matrix have?


```python
"The number of pixels by the number of nodes: 64x1"
```




    'The number of pixels by the number of nodes: 64x1'



We can set our bias term to 0: there is ony one for a singal perceptron

![sum](img/log-reg-nn-ex-sum.png)

Our inputs, the pixel, each are multiplied by their respective weights and then summed together with the bias. 

This amounts to the dotproduct of the pixel value and the weights.


# Question: Why do we have to transpose our flat_image?


```python
print(flat_image.shape)
(w.shape)

# to perform matrix multiplication, we require nxm dot mxn, 
# we need the column dimension of the left hand matrix to match the 
# to match the row dimension of the right hand matrix
```

    (64, 1)





    (64, 1)



![activation](img/log-reg-nn-ex-a.png)

Then we pass it into an activation function. The activation function converts our summed inputs into an output, which is then passed on to other nodes in hidden layers, or as an end product in the output layer. This can looslely be thought of as the action potential traveling down the axon. 



When we build our models in Keras, we will specify the activation function of both hidden layers and output.

# Question: What is an activation function we have come across? 

![don't look down](https://media.giphy.com/media/kGX9vntSO8McNlDaVj/giphy.gif)

Activation functions play the role of converting our output to a specific form. The sigmoid function converts linear equation from a number that could be any number $-\infty$ to $\infty$, to a number between 0 and 1.  This conveniently allowed us to associate the output as a probability of a certain class.

We have a suite of activation functions to choose from.

## tanh


**tanh**: $f(x) = tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

tanh a shifted version of the sigmoid. The inflection point passes through 0,0 instead of 0,.5, and the output is between -1 and 1.  This means the mean of the output is centered around 0, which can make learning in the next layer easier.  tanh is almost always better in a **hidden layer** than the sigmoid because if speeds up learning [see here](https://stats.stackexchange.com/questions/330559/why-is-tanh-almost-always-better-than-sigmoid-as-an-activation-function). For the output layer, however, sigmoid makes sense for binary outcomes.  If we require an output of 0 or 1, it makes sense for the activation function to output between 0 and 1, rather than -1 and 1.

One problem with tanh (and sigmoid), is that if our input is large, then the slope of the activation function flattens out.  When conducting backpropogation, we will use the derivative of the activation function as one of our terms multiplied by a learning rate to determine how big a step to take when adjusting our weights.  If our derivative is close to zero, the step will be very small, so the speed of our learning will be very slow, which is a huge problem.  This is called the **vanishing gradient** problem.

# ReLU

ReLU, or rectified linear unit, outputs 0 for negative numbers, and the original value for positive inputs.  

**ReLU**: $f(x) = 0$ if $x\leq 0$; $f(x) = x$ otherwise

ReLU is a commonly used and effective activation function because of speed.  Given that the **output** is zero when negative, some nodes become inactive (i.e. produce an output of 0).  Zero outputs take little computational power. Also, the constant gradient leads to faster learning in comparison to sigmoid and tanh, which come close to 0 with large positive and negative values.  Since the speed of our network is linked to the derivative, a derivative close to zero will result in very slow learning.

Notice that ReLU ("Rectified Linear Unit") increases without bound as $x\rightarrow\infty$. The advantages and drawbacks of this are discussed on [this page on stackexchange](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks)

There are many activation functions, [see here](https://towardsdatascience.com/comparison-of-activation-functions-for-deep-neural-networks-706ac4284c8a). 

Neural networks draw their inspiration from the biology of our own brains, which are of course also accurately described as 'neural networks'. A human brain contains around $10^{11}$ neurons, connected very **densely**.

![dense](img/dogcat.gif)

Our nodes will be taking in input from multiple sources. Let's add the entire training set as our input. 


Numpy allows us to easily calculate the predictions for the set of data:

### Question: What dimension should our weight vector now be?


```python
'''The same as before.  Each weight is associated with one pixel 
location across the entire training set'''
```




    'The same as before.  Each weight is associated with one pixel \nlocation across the entire training set'



### Question: What should be the dimension of the output of our collector function be?


```python
'''1437x1 one sum for every image'''
```




    '1437x1 one sum for every image'



For our DS purposes, we'll generally imagine our network to consist of only a few layers, including an input layer (where we feed in our data) an output layer (comprising our predictions). Significantly, there will also (generally) be one or more layers of neurons between input and output, called **hidden layers**.

One reason these are named hidden layers is that what their output actually represents in not really known.  The activation of node 1 of the first hidden layer may represent a sequence of pixel intensity corresponding to a horizontal line, or a group of dark pixels in the middle of a number's loop. 

![dense](img/Deeper_network.jpg)

Because we are unaware of how exactly these hidden layers are operating, neural networks are considered **black box** algorithms.  You will not be able to gain much inferential insight from a neural net.

Let's add **one** hidden layer to our network with **four** nodes.

Each of our pixels from our digit representation goes to each of our nodes, and each node has a set of weights and a bias term associated with it.




```python
'''64x4 one weight for every pixel for each node'''
```




    '64x4 one weight for every pixel for each node'



Now each of these neurons has a set of weights and a bias associated with it.

### What is the shape of this weight matrix?


```python
# 4x1
```

## Back propagation

After a certain number of data points have been passed through the model, the weights will be *updated* with an eye toward optimizing our loss function. (Thinking back to biological neurons, this is like revising their activation potentials.) Typically, this is  done  by using some version of gradient descent.

![bprop](img/BackProp_web.png)

### Loss Function

The loss function tells us how well our model performed by comparing the predictions to the actual values.

When we train our models with Keras, we will watch the loss function's progress across epochs.  A decreasing loss function will show us that our model is **improving**.

The loss function is associated with the nature of our output. In logistic regression, our output was binary, so our loss function was the negative loglikelihood, aka **cross-entropy**.

$$ \Large -\ loglikelihood = -\frac{1}{m} * \sum\limits_{i=1}^m y_i\log{p_i} + (1-y_i)\log(1-p_i) $$
    

For continuous variables, the loss function we have relied on is [MSE or MAE](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/).

Good [resource](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) on backpropogation with RMSE loss function.

Here is a good summary of different [loss functions]( https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html):
   

We not only use the the loss function to see our model is improving, we use it to update our parameters.  The gradient of the loss function is calculated in relation to each parameter of our neural net.

$$\large dw_1 = \displaystyle\frac{d\mathcal{L}(\hat y , y)}{d w_1} = \displaystyle\frac{d\mathcal{L}(\hat y , y)}{d \hat y}\displaystyle\frac{d\hat y}{dz}\displaystyle\frac{dz}{d w_1} = x_1 dz $$

Working through the Learn's Intro to Neural Networks will allow you to dive deep into the partial derivatives. For now, I will just point out that the derivative of the weight is multiplied by the derivative of our activation function, *$d\hat{y}$*.  Here you can get a glimpse of the problem with the sigmoid/tanh as an activation function for a hidden layer.  Since the derivative of the sigmoid approaches zero for very large positive or negative numbers, the update to the parameters (the partial derivative multiplied by a learning rate ($ \alpha $)) approaches zero.

$$w_1 := w_1 - \alpha dw_1$$

The speed of our neural net goes way down as a result, since the updates are so incrementally small.

For a deep dive into the fitting process, reference Chapter 11 in [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)

# Gradient Descent, Epochs, and Batches

Gradient descent can be performed in several different ways.  Unlike sklearn implimentation of linear regression, which finds the minimum of the loss with a closed form solution, neural networks move down the gradient **incrementally.**  

When we run our neural nets in Keras, we can set the hyperparameter verbose equal to 1, and we will see progress through **epochs**

![epoch](img/2014-10-28_anthropocene.png)

At the end of each epoch, **all examples** from are training set have passed through the network.

Different types of gradient descent update the parameters at different times.

### Batch Gradient Descent

The gradient is calculated across all values.  We can find the direction of the gradient, and proceed directly towards the minimum .

The weights are updated with regard to the cost at the **end of an epoch** after all training elements have passed through.

### Stochastic Gradient Descent

Updating the weights after all training examples have passed through can be detrimentally slow.  

SGD updates the weights after each training example. SGD requires less epochs to achieve quality coefficients. This speeds up gradient descent significantly [link](https://machinelearningmastery.com/gradient-descent-for-machine-learning/).

### Mini-Batch Gradient Descent

In mini-batch, we pass a batch, calculated the gradient, update the params, then proceed to the next batch.  It combines the advantages of batch and stochastic gradient descent: it is more faster than SGD since the updates are not made with each point, and more computationally efficient than batch, since all training examples don't have to fit in memory.

[Good comparison of types of Gradient Descent and batch size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

> Tip 1: A good default for batch size might be 32.  
    - batch size is typically chosen between 1 and a few hundreds, e.g. batch size = 32 is a good default value, 



# Optimizers

One of the levers we can tweek are the optimizers which control how the weights and biases are updated.

For stochastic gradient descent, the weights are updated with a **constant** learning rate (alpha) after every record.  If we specify a batch size, the constant learning rate is multiplied by the gradient across the batch. 

Other optimizers, such as **Adam** (not an acronym) update the weights in different ways. For Adam,
> A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds. [source](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) 





To be clear, backpropogation calculates the gradient for each weight and bias, in each layer, including the input layer, for each **batch**.

So, to be clear:

For mini-batch gradient descent, we:
    - pass in a specified random sample of our training set
    - the set propogates forward through our network
    - each node sums the input, adds a bias, and applies an activation function to pass to the next layer.
    - We make predictions on the output layer, then calculate the loss for back propogation.
    - We calculate the derivative of the loss with regard to each weight and bias.
    - We multiply that derivative by a learning rate determined by our optimizer.
    - We update our parameters.
    - We repeat for each batch until all examples have been used.
    - We progress to the next epoch.
 



![backprop](img/ff-bb.gif)

The graphic above can be a bit frustrating since it moves fast, but follow the progress as so:

Forward propogation with the **blue** tinted arrows computes the output of each layer: i.e. a summation and activation.

Backprop calculates the partial derivative (**green** circles) for each weight (**brown** line) and bias.

Then the optimizer multiplies a **learning rate** ($\eta$) to each partial derivative to calculate a new weight which will be applied to the next batch that passes through.
