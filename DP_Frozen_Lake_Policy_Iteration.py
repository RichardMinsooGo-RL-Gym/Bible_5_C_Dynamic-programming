# 2.1 IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np

from IPython.display import clear_output

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            gamma (float): discount factor
        """
        
        # 2.3 CREATING THE Q-TABLE
        self.env = env
        
        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        
        self.gamma = 0.9    # discount rate
        
    def one_step_lookahead(self, env, state, V, discount_factor):
        action_values = np.zeros(self.action_size)
        for action in range(self.action_size):
            for probability, next_state, reward, done in self.env.P[state][action]:
                action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values

    def policy_evaluation(self, policy, env, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Number of evaluation iterations
        evaluation_iterations = 1
        # Initialize state-value function with zeros for each env state
        V = np.zeros(self.state_size)
        # Repeat until change in value is below the threshold
        for i in range(int(max_iterations)):
            # Initialize a change of value function as zero
            delta = 0
            # Iterate though each state
            for state in range(self.state_size):
                # Initial a new value of current state
                best_action_value = 0
                # Try all possible actions which can be taken from this state
                for action, action_probability in enumerate(policy[state]):
                    # Check how good next state will be
                    for state_probability, next_state, reward, done in self.env.P[state][action]:
                        # Calculate the expected value
                        best_action_value += action_probability * state_probability * (reward + discount_factor * V[next_state])

                # Calculate the absolute change of value function
                delta = max(delta, np.abs(V[state] - best_action_value))
                
                # Update the value function for current state
                V[state] = best_action_value
            evaluation_iterations += 1

            # Terminate if value change is insignificant
            if delta < theta:
                print(f'Policy evaluated in {evaluation_iterations} iterations.')
                return V

    def policy_iteration(self, env, discount_factor=1.0, max_iterations=1e9):
        # Start with a random policy
        #num states x num actions / num actions
        policy = np.ones([self.state_size, self.action_size]) / self.action_size
        # Initialize counter of evaluated policies
        evaluated_policies = 1
        # Repeat until convergence or critical number of iterations reached
        for i in range(int(max_iterations)):
            policy_stable = True

            # Evaluate current policy
            V = self.policy_evaluation(policy, env, discount_factor=discount_factor)

            # Go through each state and try to improve actions that were taken (policy Improvement)
            for state in range(self.state_size):
                # Choose the best action in a current state under current policy
                current_action = np.argmax(policy[state])
                # Look one step ahead and evaluate if current action is optimal
                # We will try every possible action in a current state
                action_value = self.one_step_lookahead(self.env, state, V, discount_factor)
                # Select best action based on the highest state-action value
                best_action = np.argmax(action_value)
                # If action didn't change
                if current_action != best_action:
                    policy_stable = True
                    # Greedy policy update
                    policy[state] = np.eye(self.action_size)[best_action]
            evaluated_policies += 1
            # If the algorithm converged and policy is not changing anymore, then return final policy and value function
            if policy_stable:
                print(f'Evaluated {evaluated_policies} policies.')
                return policy, V

'''
Policy iteration
'''

# 2.2 CREATING THE ENVIRONMENT
env_name = "FrozenLake-v1"
env = gym.make(env_name)
env.seed(777)     # reproducible, general Policy gradient has high variance

# 2.4 INITIALIZING THE Q-PARAMETERS
max_episodes = 10000  # Set total number of episodes to train agent on.

max_iterations = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate
render = False                # display the game environment


# train
agent = DQNAgent(
    env, 
#     memory_size, 
#     batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    # Search for an optimal policy using policy iteration
    policy, V = agent.policy_iteration(env.env)
    # Apply best policy to the real env
    
    wins = 0
    episode_reward = 0
    
    for episode in range(max_episodes):
        state = agent.env.reset()
        done = False  # has the enviroment finished?
        
        if render: env.render()
            
        # 2.7 EACH TIME STEP    
        while not done:
            # Select best action to perform in a current state
            action = np.argmax(policy[state])
            # Perform an action an observe how env acted in response
            next_state, reward, done, _ = agent.env.step(action)

            if render: env.render()
            # Our new state is state
            state = next_state
            
            # Summarize total reward
            episode_reward += reward
            # Calculate number of wins over episodes
            if done and reward == 1.0:
                wins += 1
    average_reward = episode_reward / max_episodes
    
    print(f'Policy Iteration : number of wins over {max_episodes} episodes = {wins}')
    print(f'Policy Iteration : average reward over {max_episodes} episodes = {average_reward} \n\n')


