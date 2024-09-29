import bc as bc
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def evaluate(env, learner):
    NUM_TRAJS = 50
    total_learner_reward = 0
    for i in range(NUM_TRAJS):
        done = False
        obs = env.reset(seed = i)
        while not done:
            with torch.no_grad():
                hidden_obs = np.append(obs[:-1], np.array([0]))
                action = learner.get_action(hidden_obs)
            obs, reward, done, _ = env.step(action)
            total_learner_reward += reward
            if done:
                break
    return total_learner_reward / NUM_TRAJS

def interact(env, learner, expert, observations, actions, validation_obs, validation_actions, checkpoint_path, seed, num_epochs=100, horizon=1000):
    """Interact with the environment and update the learner policy using DAgger.
   
    This function interacts with the given Gym environment and aggregates to
    the BC dataset by querying the expert.
    
    Parameters:
        env (Env)
            The gym environment (in this case, the Hopper gym environment)
        learner (Learner)
            A Learner object (policy)
        expert (ExpertActor)
            An ExpertActor object (expert policy)
        observations (torch.tensor of numpy.ndarray)
            An initially empty list of numpy arrays 
        actions (torch.tensor of numpy.ndarray)
            An initially empty list of numpy arrays 
        checkpoint_path (str)
            The path to save the best performing model checkpoint
        seed (int)
            The seed to use for the environment
        num_epochs (int)
            Number of epochs to run the train function for
    """
    # Interact with the environment and aggregate your BC Dataset by querying the expert
    observations.to(device)
    NUM_INTERACTIONS = 5
    best_reward = 0
    best_model_state = None
    for episode in range(NUM_INTERACTIONS):
        # Aggregate 10 trajectories per interaction
        for _ in range(10):
            done = False
            obs = env.reset()
            for _ in range(horizon):
                # TODO: Implement Hopper environment interaction and dataset aggregation here
                # new_action = learner(obs)
                # new_obs, reward, done, _ = env.step(new_action)
                # obs.append(new_obs)
                # actions.append(expert(obs))
                with torch.no_grad():
                    hidden_obs = np.append(obs[:-1], np.array([0]))
                    learner_action = learner.get_action(hidden_obs)
                    expert_action = expert.get_expert_action(obs)

                    observations = torch.cat((observations, torch.unsqueeze(torch.tensor(hidden_obs), 0).float().to(device)))
                    actions = torch.cat((actions, torch.unsqueeze(torch.tensor(expert_action), 0).float().to(device)))

                    obs, reward, done, _ = env.step(learner_action)
                    
                if done:
                    break
            
        bc.train(learner, observations, actions, validation_obs, validation_actions, None, num_epochs)
        reward = evaluate(env, learner)
        print(f"After interaction {episode}, reward = {reward}")
        # Saving model state if current reward is greater than best reward
        if reward > best_reward:
            best_reward = reward
            best_model_state = learner.state_dict()
    # Save the best performing checkpoint
    torch.save(best_model_state, checkpoint_path)