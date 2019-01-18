# Generative Models Tutorial
Generative models are interesting topic in ML. Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data. There are different types of ways of modelling same distribution of training data: Auto-Regressive models, Auto-Encoders and GANs. In this tutorial, we are focusing theory of generative models, demonstration of generative models, important papers, courses related generative models. It will continue to be updated over time.

**Keywords:**  Generative Models, Variational Auto Encoders (VAE), Generative Adversial Networks (GAN), VAE/GAN papers, courses, etc..

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table Of Content

## What is Generative Models? <a name="whatisGM"></a>

Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data (from same distribution). In our daily life, there are huge data generated from electronic devices, computers, cameras, iot, etc. (e.g. millions of images, sentences, or sounds, etc.) In generative models, a large amount of data in some domain firstly is collected and then model is trained with this large amount of data to generate data like it. There are lots of application area generative models are used:  image denoising, inpainting, super-resolution, structured prediction, exploration, etc.. Generative models are also promising in the long term future because it has a potential power to learn the natural features of a dataset automatically. Generative models are mostly used to generate images (vision area). In the recent studies, it will also used to generate sentences (natural language processing area). 

## Unsupervised Learning vs Supervised Learning:

### Gaussian Mixture Model (GMM):

## Auto-Regressive Models:

### PixelRNN

### PixelCNN

## Variational Auto Encoder (VAE):

![vae_learning](https://user-images.githubusercontent.com/10358317/51377315-a81c3a80-1b1b-11e9-8298-7e61e6cfe329.gif)


### AutoEncoder:

### Bayesian:

## Generative Adversial Networks (GANs):

![gan_gif](https://user-images.githubusercontent.com/10358317/51377616-65a72d80-1b1c-11e9-8a7b-83c9571eac08.gif)

### DCGAN

## Generative Model in Reinforcement Learning:

### Generative Adversarial Imitation Learning:
Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

"The standard reinforcement learning setting usually requires one to design a reward function that describes the desired behavior of the agent. However, in practice this can sometimes involve expensive trial-and-error process to get the details right. In contrast, in imitation learning the agent learns from example demonstrations (for example provided by teleoperation in robotics), eliminating the need to design a reward function. This approach can be used to learn policies from expert demonstrations (without rewards) on hard OpenAI Gym environments, such as Ant and Humanoid." [Blog Open-AI]. 


## Important Papers:
Jonathan Ho, Stefano Ermon, [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

## Courses: 
[Stanford Generative Model Video](https://www.youtube.com/watch?v=5WoItGTWV54)

## References:
[Blog Open-AI](https://blog.openai.com/generative-models/#going-forward)
[PixelRNN, PixelCNN](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)

## Notes:
PixelRNN, PixelCNN: https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
