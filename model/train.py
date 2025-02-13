import os
from PIL import Image
import torch
from torch import nn
from Modelclass import DroneAcoustics

#region Load data
# REMAKE THIS PART, WE NEED TO LOAD THE DATA FROM THE CSV FILE
data_dir = './data/kaggle'
labels = []
drone_data = []
drone_far_data = []
noise_data = []

def load_data(data_dir, data_list):
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)
            data_list.append(img)

# Iterate through all files in the data directory
load_data(os.path.join(data_dir, '/drone'), drone_data)
load_data(os.path.join(data_dir, '/drone_far'), drone_far_data)
load_data(os.path.join(data_dir, '/noise'), noise_data)

data = 
#endregion

#region preprocess data

def image_preprocessing(data_list):
    processed_data = []
    for img in data_list:
        kernel = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]).float() # define kernel (edge detection)
        img = img.convolve(kernel) # apply convolution
        img = img.resize((28, 28)) # resize to 28x28
        img = img.convert('L') # convert to grayscale
        img = torch.tensor(img.getdata()).reshape(1, 28, 28) # reshape to 28x28 tensor
        img = img.to('gpu')
        img = img / 255.0 # normalize
        processed_data.append(img) 
    return processed_data

image_preprocessing(drone_data)
image_preprocessing(drone_far_data)
image_preprocessing(noise_data)

#endregion

#region Train model

model = DroneAcoustics().to('gpu')    

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('gpu'), y.to('gpu')

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#endregion
