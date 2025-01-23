
# Individual Project 3:
# Image Generation with GAN

#### Due Date
* Tuesday Apr 8, 2025 (23:59)

#### Total Points 
* 100 (One Hundred)

## Goal
In this assignment you will be asked to implement a Generative Adversarial Networks (GAN) with [MNIST data set](http://yann.lecun.com/exdb/mnist/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/tutorials/). 

<img src="https://github.com/yanhuata/DS504CS586-S20/blob/master/project3/pic/goal.png" width="80%">


## Project Guidelines

#### Data set

MNIST is a dataset composed of handwritten numbers and their labels. Each MNIST image is a 28\*28 grey-scale image, which is labeled with an integer value between 0 and 9, corresponding to the actual value in the image. MNIST is provided in Pytorch as 28\*28 matrices containing numbers ranging from 0 to 255. There are 60000 images and labels in the training data set and 10000 images and labels in the test data set. Since this project is an unsupervised learning project, you can only use the 60000 images for your training. 

Download MNIST dataset directly in Pytorch:

```python
from torchvision import datasets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
```

#### Installing Software and Dependencies 

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Create virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Install packages (e.g. pip install torch)

#### Building and Compiling Generator and Discriminator

In Pytorch, you can try different layers, such as “Conv2D”, and different activation functions, such as “tanh”, and “leakyRelu”. You can apply different optimizers, such as stochastic gradient descent or Adam, and different loss functions. The following is the sample code of how to build the model.


```python
# Create a Generator class.
class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Generator.
netG = Generator(*args)

# Create a Discriminator class.
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Discriminator.
netD = Discriminator(*args)

# Setup Generator optimizer.
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Setup Discriminator optimizer.
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Define loss function. 
criterion = torch.nn.BCELoss()
```

#### Training GAN

You have the option of changing how many epochs to train your model for and how large your batch size is. The following is the sample code of how to train GAN. You can add self-defined parameters such as #epoch, learning rate scheduler to the train function.


```python
# Training
def train():
    for _ in range(batchCount):  
	
        # Create a batch by drawing random index numbers from the training set
       
        # Create noise vectors for the generator
        
        # Generate the images from the noise

        # Create labels

        # Train discriminator on generated images

        # Train generator

```

#### Saving Generator

Please use the following code to save the model and weights of your generator.



```python
# save model with Pytorch
torch.save(netG, './generator.pt')
torch.save(netG.state_dict(), './generator_weights.pt')
```

#### Plotting

Please use the following code to plot the generated images. As for the loss plot of your generator and discriminator during the training, you can plot with your own style. 


```python
# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.random.normal(size=[num_gen, z_dim])
generated_images = netG(z)

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
plt.show()
```

## Deliverables

### Gradescope Assignment Submission Guide

Follow these instructions to correctly prepare and submit your assignment on Gradescope.

---

### How to Submit Your Assignment on Gradescope

#### 1. Train Your Model & Save It

Train your model.
Save the trained generator to a file named **`generate.pt`**.

```python
torch.save(model, "generate.pt")
```

#### 2. Provide Your Generator Definition
Ensure your code that defines the Generator class is in a file called training.py, with a class signature like this:
```python
class Generator(nn.Module):
    # Your implementation here
```

#### 3. Optional: Additional Requirements
If your solution depends on any additional Python packages not included in the base grader environment (e.g., custom libraries or specific versions), list them in a requirements.txt file.
The Autograder will automatically install any dependencies specified in this file.
Example requirements.txt:
```python
torch==2.0.1
numpy==1.23.5
matplotlib==3.7.1
```
#### 4. Zip Your Submission

Prepare a ZIP file containing the following files:

- **`generate.pt`**: Your trained model file.
- **`training.py`**: Your `Generator` class definition.
- (Optional) **`requirements.txt`**: If extra dependencies are needed.
- **PDF Report**: Include a detailed report with the following sections:

#### PDF Report Contents

1. **Set of Experiments Performed**  
   - Describe the set of experiments you performed:
     - What structures you experimented with (e.g., number of layers, number of neurons in each layer).
     - What hyperparameters you vary (e.g., number of epochs, batch size, learning rate).
     - Any other parameter values you experimented with.
   - Include details about the weight initialization schema, activation functions, loss functions, and optimizers you used.

2. **Special Skills**  
   - Include any skills or methods you used to improve generation quality.
   - Refer to [GAN Hacks](https://github.com/soumith/ganhacks) for tips and include anything relevant you applied.

3. **Visualization**  
   - Include:
     - **25 (5×5) final generated images**: Format them as shown in the assignment goal.
       - Ensure you generate at least **one image for each digit (0–9)**.
     - **Loss plot**: Show the generator and discriminator losses during training.

Ensure the ZIP file contains all the required files (`generate.pt`, `training.py`, and your PDF report). Missing files may result in deductions or incomplete grading.

- Important:
- Ensure the file names match exactly (generate.pt and training.py).
- Do not include files like evaluation.py.

## Grading

#### Report (70%)

* Set of experiments performed: 30 points
* Special skills: 20 points
* Visualization: 20 points

#### Code (20%) 

To achieve full marks for the coding portion of your assignment, you must pass Autograder, Autograder will save the plot you generated.

###### Criteria for Full Credits:
* Code Clarity: Ensure your code is well-commented and organized.

#### Model (10%)

You can get full credits if all the generated images can be recognized, otherwise, you may lose some points. Also, the code you submitted should be able to generate all 10 different digits.

## Bonus (10 points)

Generate images from other data sources or train a variation of basic GAN model e.g., cGAN, CycleGAN, as we introduced in the lecture), please highlight it in the report.

* Data set

  Here are some images you may be interested in. Note other data sets are also allowed.
  
  [Face](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)
  
  [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
  
  [Anime](https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I)
  
* Package

  You can use any deep learning package, such as Pytorch, etc.
  
* Deliverable

  * Code
  
  * Model
  
  * README file (How to compile and load the model to generate images)
  
  * 25 generated images

## Tips for Using GPU on Turing Server

* Follow Turing_Setup_Instruction.pdf
* Submit job on Turing server
   ``` shell
   #!/bin/bash
   #SBATCH -A ds504
   #SBATCH -p academic
   #SBATCH -N 1
   #SBATCH -c 8
   #SBATCH --gres=gpu:1
   #SBATCH -t 12:00:00
   #SBATCH --mem 12G
   #SBATCH --job-name="p3"
   
   python evaluation.py
   ```

