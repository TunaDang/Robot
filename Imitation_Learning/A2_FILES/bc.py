import gym
import tqdm
from tqdm import tqdm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import optimizer

def train(learner, observations, actions, validation_obs, validation_acts, checkpoint_path, num_epochs=100):
    """Train function for learning a new policy using BC.
    
    Parameters:
        learner (Learner)
            A Learner object (policy)
        observations (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 11, ) 
        actions (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 3, )
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            Number of epochs to run the train function for
    
    Returns:
        learner (Learner)
            A Learner object (policy)
    """
    best_loss = float('inf')
    best_model_state = None

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    dataset = TensorDataset(observations, actions) # Create your dataset
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True) # Create your dataloader

    # TODO: Complete the training loop here ###
    loss = float('inf')
    validation_losses = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        # TODO: Iterate through dataloader and train

        # TODO: Calculate Validation Loss and Append to validation_losses (remember to use "with torch.no_grad():")


        # Saving model state if current loss is less than best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = learner.state_dict()

    # Save the best performing checkpoint
    if checkpoint_path:
        torch.save(best_model_state, checkpoint_path)
        # Save graph if not running DAGGER:
        plt.plot(np.arange(0, num_epochs), [t.cpu().numpy() for t in validation_losses])
        plt.title("Validation Loss vs. Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.savefig("BC_validation_loss.png")
    
    return learner