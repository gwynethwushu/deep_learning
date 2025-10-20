import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim

# build autoencoder
class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim) # input layer
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim) # batch normalization 
        self.input_dropout = nn.Dropout(dropout_rate) # dropout
        
        self.hidden_layer = nn.Linear(hidden_dim, latent_dim) # hidden layer
        self.output_layer = nn.Linear(latent_dim, input_dim) # output layer

    def encode(self, x):
        # apply activation and dropout
        x = self.input_dropout(x) # randomly set connection weights to 0
        
        x = self.input_layer(x) # encode input layer to the hidden layer
        x = self.batch_norm(x) # batch normalization
        x = torch.relu(x) # ReLu activation

        x = self.hidden_layer(x) # encode hidden layer to latent layer
        x = torch.relu(x) # ReLu activation

        return x
    
    def decode(self, z):
        # decode latent representation back to input space
        z = self.output_layer(z) # decode latent layer to output layer
        return z
    
    def forward(self, x):
        z = self.encode(x) # encode the input
        recon = self.output_layer(z) # decode the latent representation
        return recon
    
    def inference(self, x):
        self.eval() # switch to evaluation mode (disables dropout and BatchNorm uses running stats collected during training) 
        with torch.no_grad(): # disables gradient tracking
            z = self.encode(x) # encode the input
            recon = self.output_layer(z) # decode the latent representation
            recon = torch.clamp(recon, min=-1.0, max=1.0) # force values from output layer to [-1, 1]
            return recon
        
def compute_loss(true, pred):
    # compute loss between the true and predicted values. MSE used here.
    loss = torch.mean((true - pred) ** 2)
    return loss

def train_step(model, data, optimizer):
    # perform one training step for the model
    model.train() # set the model to training mode (enables dropout)
    optimizer.zero_grad() # clear previous gradients
        
    recon = model.forward(data) # forward pass
    loss = compute_loss(data, recon) # compute loss 
    loss.backward() # backpropagation
    optimizer.step() # update the weights
    return loss.item()