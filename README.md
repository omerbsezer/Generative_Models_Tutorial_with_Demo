# Generative Models Tutorial
Generative models are interesting topic in ML. Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data. There are different types of ways of modelling same distribution of training data: Auto-Regressive models, Auto-Encoders and GANs. In this tutorial, we are focusing theory of generative models, demonstration of generative models, important papers, courses related generative models. It will continue to be updated over time.

<p align="center">
<img src="https://user-images.githubusercontent.com/10358317/51377315-a81c3a80-1b1b-11e9-8298-7e61e6cfe329.gif">[Blog Open-AI]
</p>

**Keywords:**  Generative Models, Variational Auto Encoders (VAE), Generative Adversial Networks (GAN), VAE/GAN papers, courses, etc..

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table Of Content

## What is Generative Models? <a name="whatisGM"></a>
- Generative models are a subset of unsupervised learning that generate new sample/data by using given some training data (from same distribution). In our daily life, there are huge data generated from electronic devices, computers, cameras, iot, etc. (e.g. millions of images, sentences, or sounds, etc.) 
- In generative models, a large amount of data in some domain firstly is collected and then model is trained with this large amount of data to generate data like it. 
- There are [lots of applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900) for generative models:  
  - Image Denoising, 
  - Image Enhancement, 
  - [Image Inpainting](https://github.com/pathak22/context-encoder), 
  - [Super-resolution (upsampling): SRGAN](https://arxiv.org/pdf/1609.04802.pdf), 
  - [Generate 3D objects: 3DGAN](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf), [Video](https://youtu.be/HO1LYJb818Q) 
  - Creating Art:
     - [Create Anime Characters](https://arxiv.org/pdf/1708.05509.pdf)
     - [Transform images from one domain (say real scenery) to another domain (Monet paintings or Van Gogh):CycleGAN](https://github.com/junyanz/CycleGAN)
     - [Creating Emoji: DTN](https://arxiv.org/pdf/1611.02200.pdf)
  - [Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)
  - [Creating clothing images and styles from an image: PixelDTGAN](https://arxiv.org/pdf/1603.07442.pdf)
  - [Face Synthesis: TP-GAN](https://arxiv.org/pdf/1704.04086.pdf)
  - [Image-to-Image Translation: Pix2Pix](https://github.com/phillipi/pix2pix)
      - Labels to Street Scene
      - Aerial to Map
      - Sketch to Realistic Image
  - [High-resolution Image Synthesis](https://tcwang0509.github.io/pix2pixHD/)
  - [Text to image: StackGAN](https://github.com/hanzhanggit/StackGAN), [Paper](https://arxiv.org/pdf/1612.03242v1.pdf)
  - [Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)
  - [Learn Joint Distribution: CoGAN](https://arxiv.org/pdf/1606.07536.pdf)
  - [Transfering style (or patterns) from one domain (handbag) to another (shoe): DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch)
  - [Texture Synthesis: MGAN](https://arxiv.org/pdf/1604.04382.pdf)
  - [Image Editing: IcGAN](https://github.com/Guim3/IcGAN)
  - [Face Aging: Age-cGAN](https://arxiv.org/pdf/1702.01983.pdf)
  - [Neural Photo Editor](https://github.com/ajbrock/Neural-Photo-Editor)
  - [Medical (Anomaly Detection): AnoGAN](https://arxiv.org/pdf/1703.05921.pdf)
  - [Music Generation: MidiNet](https://arxiv.org/pdf/1703.10847.pdf)
  - [Video Generation](https://youtu.be/Pt1W_v-yQhw)
  - [Image Blending: GP-GAN](https://github.com/wuhuikai/GP-GAN)
  - [Object Detection: PerceptualGAN](https://arxiv.org/pdf/1706.05274v2.pdf)
  - Natural Language:
      - Artical Spinning
      
- Generative models are also promising in the long term future because it has a potential power to learn the natural features of a dataset automatically. 
- Generative models are mostly used to generate images (vision area). In the recent studies, it will also used to generate sentences (natural language processing area). 
- It can be said that Generative models begins with sampling. Generative models are told in this tutorial according to the development steps of generative models: Sampling, Gaussian Mixture Models, Variational AutoEncoder, Generative Adversial Networks.   

## Preliminary (Recall)
- **Bayesian Rule**: p(z|x)= p(x|z) p(z) /p(x)
- **Prior Distribution**: "often simply called the prior, of an uncertain quantity is the probability distribution that would express one's beliefs about this quantity before some evidence is taken into account." (e.g. p(z)) 
- **Posterior Distribution:** is a probability distribution that represents your updated beliefs about the parameter after having seen the data. (e.g. p(z|x))
- **Posterior probability = prior probability + new evidence (called likelihood)**
- **Probability Density Function (PDF)**: the set of possible values taken by the random variable
- **Gaussian (Normal) Distribution**: A symmetrical data distribution, where most of the results lie near the mean.
- **Bayesian Analysis**:
  - Prior distribution: p(z)
  - Gather data
  - "Update your prior distribution with the data using Bayes' theorem to obtain a posterior distribution."
  - "Analyze the posterior distribution and summarize it (mean, median, etc.)"
- It is expected that you have knowledge of neural network concept (gradient descent, cost function, activation functions, regression, classification)
  - Typically used for regression or classification
  - Basically: fit(X,Y) and predict(X)

## Sampling from Bayesian Classifier
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

## Unsupervised Deep Learning vs Supervised Deep Learning (Recall)
- **Unsupervised Deep Learning**: trying to learn structure of inputs 
  - **Example1**: If the structure of poetry/text can be learned, it is possible to generate text/poetry that resembles the given text/poetry.
  - **Example2**: If the structure of art can be learned, it is possible to make new art/drawings that resembles the given art/drawings.
  - **Example3**: If the structure of music can be learned, it is possible to create new music that resembles the given music.
- **Supervised Deep Learning**: trying to map inputs to targets

## Gaussian Mixture Model (GMM)
- Single gaussian model learns blurry images if there are more than one gaussian distribution (e.g. different types of writing digits in handwriting).
- To get best result, GMM have to used to model more than one gaussian distribution.
- GMM is latent variable model. With GMM, multi-modal distribution can be modelled at the same time.
- Multiple gaussians in different proportions are fitted into the GMM. 
- 2 clusters: p(x)=p(z=1) p(x|z=1) + p(z=2) p(x|z=2). In figure, there are 2 different proportions gaussian distributions.

<p align="center">
<img src="https://user-images.githubusercontent.com/10358317/51385984-e9b7e000-1b31-11e9-8d7e-df4f3dc72d4f.png">[Udemy GAN-VAE]
</p>

### Expectation-Maximization (EM)
- GMM is trained using Expectation-Maximization (EM)
- EM is iterative algorithm that let the likelihood improves at each step.
- The aim of EM is to reach maximum likelihood.

## AutoEncoders
- A neural network that predicts (reconstructs) its own input.
- It is a feed forward network.
- W: weight, b:bias, x:input, f() and g():activation functions, z: latent variable, x_hat= output (reconstructed input)

<p align="center">
<img src="https://user-images.githubusercontent.com/10358317/51426045-ed149f80-1bf5-11e9-8d7f-5185e60139c5.png">
</p>

- Instead of fit(X,Y) like neural networks, autoencoders fit(X,X).
- It has 1 input, 1 hidden, 1 output layers (hidden layer size < input layer size; input layer size = output layer size)
- It forces neural network to learn compact/efficient representation (e.g. dimension reduction/ compression)

<p align="center">
<img src="https://user-images.githubusercontent.com/10358317/51418718-2c130880-1b96-11e9-9e2c-41fcd15da4b0.png">[Udemy GAN-VAE]
</p>

## Variational Inference (VI)
- Variational inference (VI) is the significant component of Variational AutoEncoders.
- VI ~ Bayesian extension of EM.
- In GMM/K-Means Clustering, you have choose the number of clusters.
- VI-GMM (Variational inference-Gaussian Mixture Model) automatically finds the number of cluster.

![variational-inference](https://user-images.githubusercontent.com/10358317/51386321-128ca500-1b33-11e9-8367-ea8c73e305c1.png)
[Udemy GAN-VAE]

## Variational AutoEncoder (VAE)
- VAE is a neural network that learns to generate its input.
- It can map data to latent space, then generate samples using latent space.
- VAE is combination of autoencoders and variational inference. 
- It doesn't work like tradional autoencoders. 
- Its output is the parameters of a distribution: mean and variance, which represent a Gaussian-PDF of Z  (instead only one value)
- VAE consists of two units: Encoder, Decoder. 
- The output of encoder represents Gaussian distributions.(e.g. In 2-D Gaussian, encoder gives 2 mean and 2 variance/stddev)
- The output of decoder represents Bernoulli distributions.
- From a probability distribution, new samples can be generated.
- For example: let's say input x is a 28 by 28-pixel photo
- The encoder ‘encodes’ the data which is 784-dimensional into a latent (hidden) representation space z.
- The encoder outputs parameters to q(z∣x), which is a Gaussian probability density.
- The decoder gets as input the latent representation of a digit z and outputs 784 Bernoulli parameters, one for each of the 784 pixels in the image. 
- The decoder ‘decodes’ the real-valued numbers in z into 784 real-valued numbers between 0 and 1.

![vae1](https://user-images.githubusercontent.com/10358317/51426260-f0f5f100-1bf8-11e9-98e5-b8bbf4e3cf22.png) [Udemy GAN-VAE]

### Latent Space
- Encoder: x-> q(z) {q(z): latent space, coded version of x}
- Decoder: q(z)~z -> x_hat
- Encoder takes the input of image "8" and gives output q(z|x).
- Sample from q(z|x) to get z
- Get p(x_hat|x), sample from it (this is called posterior predictive sample)

![latent_space](https://user-images.githubusercontent.com/10358317/51431824-861bd880-1c3f-11e9-8951-ecf9ce3f08a7.png)


### Cost Function of VAE
- Evidence Lower Bound (ELBO) is our objective function that has to be maximized.


## Generative Adversial Networks (GANs)
- There are 2 different networks: generator and discriminator, compete against each other.
- GANs are interesting because it generates samples exceptionally good.

![gan_gif](https://user-images.githubusercontent.com/10358317/51377616-65a72d80-1b1c-11e9-8a7b-83c9571eac08.gif) [Blog Open-AI]

### DCGAN

### CycleGAN
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

![cyclegan](https://user-images.githubusercontent.com/10358317/51417525-704eda80-1b8f-11e9-93ce-2d3c14a3aee1.jpeg)

### Pix2Pix
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

![pix2pix](https://user-images.githubusercontent.com/10358317/51417511-6e851700-1b8f-11e9-84ed-64cfb0cd6e58.png)

### PixelDTGAN
[Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442.pdf)

![pixelgan](https://user-images.githubusercontent.com/10358317/51417512-6e851700-1b8f-11e9-8557-003e9c4e9ec5.png)

### PoseGuided
[Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)

![poseguided](https://user-images.githubusercontent.com/10358317/51417513-6e851700-1b8f-11e9-90b8-314377157b6f.png)

### SRGAN
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)

![srgan](https://user-images.githubusercontent.com/10358317/51417515-6f1dad80-1b8f-11e9-9312-ee6f02e6f4f5.png)

### StackGAN
[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v1.pdf)

![stackgan](https://user-images.githubusercontent.com/10358317/51417516-6f1dad80-1b8f-11e9-9309-d44ca62983c7.png)

### TPGAN
[Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](https://arxiv.org/pdf/1704.04086.pdf)

![tp-gan](https://user-images.githubusercontent.com/10358317/51417517-6f1dad80-1b8f-11e9-89ea-09fdaa77d794.png)

### Anime Generation
[Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/pdf/1708.05509.pdf)

![animegeneration](https://user-images.githubusercontent.com/10358317/51417518-6fb64400-1b8f-11e9-8205-58d408e3556c.png)

### 3DGAN
[Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)

![3dgan](https://user-images.githubusercontent.com/10358317/51417520-6fb64400-1b8f-11e9-8f09-806b6636a13b.png)

### Age-cGAN
[FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1702.01983.pdf)

![age-cgan](https://user-images.githubusercontent.com/10358317/51417523-6fb64400-1b8f-11e9-8e7b-80fe044a6586.png)

### AnoGAN
[Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/pdf/1703.05921.pdf)

![anogan](https://user-images.githubusercontent.com/10358317/51417524-704eda80-1b8f-11e9-914c-79ed3c528c21.png)

### DiscoGAN
[Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf)

![discogan](https://user-images.githubusercontent.com/10358317/51417526-70e77100-1b8f-11e9-8e2d-d25891a02ff3.png)

### DTN
[UNSUPERVISED CROSS-DOMAIN IMAGE GENERATION](https://arxiv.org/pdf/1611.02200.pdf)

![dtn](https://user-images.githubusercontent.com/10358317/51417527-70e77100-1b8f-11e9-8e0b-9cc45fa2d632.png)

### IcGAN
[Invertible Conditional GANs for image editing](https://arxiv.org/pdf/1611.06355.pdf)

![icgan](https://user-images.githubusercontent.com/10358317/51417528-70e77100-1b8f-11e9-94d9-3f0c91cf123a.png)

### MGAN
[Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/pdf/1604.04382.pdf)

![mgan](https://user-images.githubusercontent.com/10358317/51417529-71800780-1b8f-11e9-9834-a4812af4518f.png)

### MidiNet
[MIDINET: A CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORK FOR SYMBOLIC-DOMAIN MUSIC GENERATION](https://arxiv.org/pdf/1703.10847.pdf)

![midinet](https://user-images.githubusercontent.com/10358317/51417530-71800780-1b8f-11e9-801b-896122c13614.png)

### PerceptualGAN
[Perceptual Generative Adversarial Networks for Small Object Detection](https://arxiv.org/pdf/1706.05274v2.pdf)

![perceptualgan](https://user-images.githubusercontent.com/10358317/51417531-71800780-1b8f-11e9-8df1-0bb1befe4d9c.png)


## Generative Model in Reinforcement Learning:

### Generative Adversarial Imitation Learning:
Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

"The standard reinforcement learning setting usually requires one to design a reward function that describes the desired behavior of the agent. However, in practice this can sometimes involve expensive trial-and-error process to get the details right. In contrast, in imitation learning the agent learns from example demonstrations (for example provided by teleoperation in robotics), eliminating the need to design a reward function. This approach can be used to learn policies from expert demonstrations (without rewards) on hard OpenAI Gym environments, such as Ant and Humanoid." [Blog Open-AI]. 

![running_human](https://user-images.githubusercontent.com/10358317/51384409-4cf34380-1b2d-11e9-9aa5-cf8807309e73.gif) [Blog Open-AI]


## Auto-Regressive Models

### PixelRNN

### PixelCNN

## Important Papers
- Jonathan Ho, Stefano Ermon, [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- Ledig et al., [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf), 
- Jiajun Wu et al., [Learning a Probabilistic Latent Space of Object
Shapes via 3D Generative-Adversarial Modeling](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)
- Jin et al., [Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/pdf/1708.05509.pdf)
- Zhu et al., [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
- Taigman et al., [UNSUPERVISED CROSS-DOMAIN IMAGE GENERATION](https://arxiv.org/pdf/1611.02200.pdf)
- MA et al., [Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)
- Yoo et al., [Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442.pdf)
- Huang aet al., [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](https://arxiv.org/pdf/1704.04086.pdf)
- Isola et al., [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- Wang et al., [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
- Zhang et al., [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v1.pdf)
- Reed et al., [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)
- Liu et al., [Coupled Generative Adversarial Networks](https://arxiv.org/pdf/1606.07536.pdf)
- Kim et al., [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf)
- Li et al., [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/pdf/1604.04382.pdf)
- Perarnau et al., [Invertible Conditional GANs for image editing](https://arxiv.org/pdf/1611.06355.pdf)
- Antipov et al., [FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1702.01983.pdf)
- Schlegl et al., [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/pdf/1703.05921.pdf)
- Yang et al., [MIDINET: A CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORK FOR SYMBOLIC-DOMAIN MUSIC GENERATION](https://arxiv.org/pdf/1703.10847.pdf)
- Wu et al., [GP-GAN: Towards Realistic High-Resolution Image Blending](https://arxiv.org/pdf/1703.07195.pdf)
- Li et al., [Perceptual Generative Adversarial Networks for Small Object Detection](https://arxiv.org/pdf/1706.05274v2.pdf)


## Courses
- [Stanford Generative Model Video](https://www.youtube.com/watch?v=5WoItGTWV54)
- [Udemy GAN-VAE: Deep Learning GANs and Variational Autoencoders](https://www.udemy.com/deep-learning-gans-and-variational-autoencoders/learn/v4/t/lecture/7494546?start=0)

## References
- [Blog Open-AI](https://blog.openai.com/generative-models/#going-forward)
- [PixelRNN, PixelCNN](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173)
- [Udemy GAN-VAE: Deep Learning GANs and Variational Autoencoders](https://www.udemy.com/deep-learning-gans-and-variational-autoencoders/learn/v4/t/lecture/7494546?start=0)
- [GAN Applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
- https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

## Notes
- [GMM-Scikit Learn Library](https://scikit-learn.org/stable/modules/mixture.html)
- PixelRNN, PixelCNN: https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
