# Generative Models Tutorial
Generative models are interesting topic in ML. Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data. There are different types of ways of modelling same distribution of training data: Auto-Regressive models, Auto-Encoders and GANs. In this tutorial, we are focusing theory of generative models, demonstration of generative models, important papers, courses related generative models. It will continue to be updated over time.

**Keywords:**  Generative Models, Variational Auto Encoders (VAE), Generative Adversial Networks (GAN), VAE/GAN papers, courses, etc..

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table Of Content

## What is Generative Models? <a name="whatisGM"></a>

Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data (from same distribution). In our daily life, there are huge data generated from electronic devices, computers, cameras, iot, etc. (e.g. millions of images, sentences, or sounds, etc.) In generative models, a large amount of data in some domain firstly is collected and then model is trained with this large amount of data to generate data like it. There are lots of application area generative models are used:  image denoising, inpainting, super-resolution, structured prediction, exploration, etc.. Generative models are also promising in the long term future because it has a potential power to learn the natural features of a dataset automatically. Generative models are mostly used to generate images (vision area). In the recent studies, it will also used to generate sentences (natural language processing area). 

It can be said that Generative models begins with sampling. Generative models are told in this tutorial according to the development steps of generative models: Sampling, Gaussian Mixture Models, Variational AutoEncoder, Generative Adversial Networks.   

## Preliminary (Recall):
- **Bayesian Rule**: p(z|x)= p(x|z) p(z) /p(x)
- **Prior Distribution**: "often simply called the prior, of an uncertain quantity is the probability distribution that would express one's beliefs about this quantity before some evidence is taken into account." (e.g. p(z)) 
- **Posterior Distribution:** is a probability distribution that represents your updated beliefs about the parameter after having seen the data. (e.g. p(z|x))
- **Posterior probability = prior probability + new evidence (called likelihood)**
- **Bayesian Analysis**:
  - Prior distribution: p(z)
  - Gather data
  - "Update your prior distribution with the data using Bayes' theorem to obtain a posterior distribution."
  - "Analyze the posterior distribution and summarize it (mean, median, etc.)"

## Sampling from Bayesian Classifier:
- We use sampling data to generate new samples (using distribution of the training data).
- If we know the probability distribution of the training data , we can sample from it.
- Two ways of sampling:
  - **First Method**: Sample from a given digit/class
    - Pick a class, e.g. y=2
    - If it is known p(x|y=2) is Gaussian
    - Sample from this Gaussian using Scipy (mvn.rvs(mean,cov))
  - **Second Method**: Sample from p(y)p(x|y) => p(x,y)
    - If there is graphical model (e.g. y-> x), and they have different distribution,
    - If p(y) and p(x|y) are known and y has its own distribution (e.g. categorical or discrete distribution)
    - Sample 

## Unsupervised Deep Learning vs Supervised Deep Learning (Recall):
- **Unsupervised Deep Learning**: trying to learn structure of inputs 
  - **Example1**: If the structure of poetry/text can be learned, it is possible to generate text/poetry that resembles the given text/poetry.
  - **Example2**: If the structure of art can be learned, it is possible to make new art/drawings that resembles the given art/drawings.
  - **Example3**: If the structure of music can be learned, it is possible to create new music that resembles the given music.
- **Supervised Deep Learning**: trying to map inputs to targets

## Gaussian Mixture Model (GMM):
- [GMM-Scikit Learn Library](https://scikit-learn.org/stable/modules/mixture.html)
- Single gaussian model learns blurry images if there are more than one gaussian distribution (e.g. different types of writing digits in handwriting).
- To get best result, GMM have to used to model more than one gaussian distribution.
- GMM is latent variable model.
- With GMM, multi-modal distribution can be modelled at the same time.
- Multiple gaussians in different proportions are fitted into the GMM. 



- 2 clusters: p(x)=p(z=1) p(x|z=1) + p(z=2) p(x|z=2). In figure, there are 2 different proportions gaussian distributions.

![gmm](https://user-images.githubusercontent.com/10358317/51385984-e9b7e000-1b31-11e9-8d7e-df4f3dc72d4f.png) [Udemy GAN-VAE]

### Expectation-Maximization (EM):
- GMM is trained using Expectation-Maximization (EM)
- EM is iterative algorithm that let the likelihood improves at each step.
- The aim of EM is to reach maximum likelihood.


## Variational AutoEncoder (VAE):

- Variational inference (VI) is the significant component of Variational AutoEncoders.
- VI ~ Bayesian extension of EM.
- In GMM/K-Means Clustering, you have choose the number of clusters.
- VI-GMM (Variational inference-Gaussian Mixture Model) automatically finds the number of cluster.

![variational-inference](https://user-images.githubusercontent.com/10358317/51386321-128ca500-1b33-11e9-8367-ea8c73e305c1.png)
[Udemy GAN-VAE]

![vae_learning](https://user-images.githubusercontent.com/10358317/51377315-a81c3a80-1b1b-11e9-8298-7e61e6cfe329.gif) [Blog Open-AI]


### AutoEncoder:

### Bayesian:

## Generative Adversial Networks (GANs):
- There are 2 different networks: generator and discriminator, compete against each other 

![gan_gif](https://user-images.githubusercontent.com/10358317/51377616-65a72d80-1b1c-11e9-8a7b-83c9571eac08.gif) [Blog Open-AI]

### DCGAN

## Generative Model in Reinforcement Learning:

### Generative Adversarial Imitation Learning:
Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

"The standard reinforcement learning setting usually requires one to design a reward function that describes the desired behavior of the agent. However, in practice this can sometimes involve expensive trial-and-error process to get the details right. In contrast, in imitation learning the agent learns from example demonstrations (for example provided by teleoperation in robotics), eliminating the need to design a reward function. This approach can be used to learn policies from expert demonstrations (without rewards) on hard OpenAI Gym environments, such as Ant and Humanoid." [Blog Open-AI]. 

![running_human](https://user-images.githubusercontent.com/10358317/51384409-4cf34380-1b2d-11e9-9aa5-cf8807309e73.gif) [Blog Open-AI]


## Auto-Regressive Models:

### PixelRNN

### PixelCNN

## Important Papers:
- Jonathan Ho, Stefano Ermon, [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

## Courses: 
- [Stanford Generative Model Video](https://www.youtube.com/watch?v=5WoItGTWV54)

## References:
- [Blog Open-AI](https://blog.openai.com/generative-models/#going-forward)
- [PixelRNN, PixelCNN](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)
- [Udemy GAN-VAE: Deep Learning GANs and Variational Autoencoders](https://www.udemy.com/deep-learning-gans-and-variational-autoencoders/learn/v4/t/lecture/7494546?start=0)

## Notes:
- PixelRNN, PixelCNN: https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
