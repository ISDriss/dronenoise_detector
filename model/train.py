import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Modelclass import DroneAcoustics
from Datasetclass import ImageDataset

#region Load data

# Load the CSV file (annotations file)
# csv_file = 'data/kaggle_data_labels.csv'
# data_df = pd.read_csv(csv_file)

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
print(f"Loaded {len(drone_data)} drone spectograms.")

load_data(os.path.join(data_dir, '/drone_far'), drone_far_data)
print(f"Loaded {len(drone_far_data)} drone_far spectograms.")

load_data(os.path.join(data_dir, '/noise'), noise_data)
print(f"Loaded {len(noise_data)} noise spectograms.")

data_df = pd.DataFrame({
    'img': drone_data + drone_far_data + noise_data,
    'label': ['drone']*len(drone_data) + ['drone_far']*len(drone_far_data) + ['noise']*len(noise_data)
})

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

#endregion

#region preprocess data

def image_preprocessing(image_list):
    processed_data = []
    for img in image_list:
        kernel = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]).float() # define kernel (edge detection)
        img = img.convolve(kernel) # apply convolution
        img = img.resize((28, 28)) # resize to 28x28
        img = img.convert('L') # convert to grayscale
        img = torch.tensor(img.getdata()).reshape(1, 28, 28) # reshape to 28x28 tensor
        img = img / 255.0 # normalize
        img = img.to('cpu') # move to CPU
        processed_data.append(img) 
    return processed_data

# Preprocess the training and testing data
train_processed_data = image_preprocessing(train_df['img'])
test_processed_data = image_preprocessing(test_df['img'])

# Update the DataFrames with the processed data
train_df['img'] = train_processed_data
test_df['img'] = test_processed_data

#endregion

#region Train model

batch_size = 64

# Create dataloaders for training and testing
train_dataset = ImageDataset(train_df['img'], train_df['label'])
test_dataset = ImageDataset(test_df['img'], test_df['label'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = DroneAcoustics().to('cpu')    

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to('cpu')

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

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")