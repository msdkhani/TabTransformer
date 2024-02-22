# import libraries for transformer and embedding with pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import torch.nn as nn

def embedding_layer(input,num_embeddings, embedding_dim, max_norm=None):

    embedding = nn.Embedding(num_embeddings, embedding_dim, max_norm=True)   
    Embeding = embedding(input)
    data_Tensor = torch.LongTensor(input.values)
    Embeding = embedding(data_Tensor)
    
    return Embeding

# positional encoding
def positional_encoding(d_model, max_length=5000,n=10000):
    # generate an empty matrix for the positional encodings (pe)
    pe = np.zeros(max_length*d_model).reshape(max_length, d_model) 
    # for each position
    for k in np.arange(max_length):

        # for each dimension
        for i in np.arange(d_model//2):
            # calculate the internal value for sin and cos
            theta = k / (n ** ((2*i)/d_model))       

            # even dims: sin   
            pe[k, 2*i] = math.sin(theta) 

            # odd dims: cos               
            pe[k, 2*i+1] = math.cos(theta)
    # convert the numpy array to a tensor
    pe = torch.tensor(pe, dtype=torch.float32)
    return pe

