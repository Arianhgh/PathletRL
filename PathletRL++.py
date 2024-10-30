import json
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import torch
import argparse
# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Pathlet DQN Trainer with customizable parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--M", type=float, default=0.25, help="Trajectory loss threshold")
parser.add_argument("--mu_threshold", type=float, default=0.8, help="Coverage threshold")
parser.add_argument("--k", type=int, default=10, help="Maximum pathlet length")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for DQN agent")
parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
parser.add_argument("--learning_starts", type=int, default=1000, help="Steps before training starts")
parser.add_argument("--exploration_initial_eps", type=float, default=1.0, help="Initial epsilon for exploration")
parser.add_argument("--exploration_final_eps", type=float, default=0.001, help="Final epsilon for exploration")
parser.add_argument("--exploration_fraction", type=float, default=0.6, help="Fraction of steps to decay epsilon")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train the agent")
args = parser.parse_args()
# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Load datasets
with open('data/traj_edge_dict.json', 'r') as f:
    traj_edge_dict = json.load(f)

with open('data/pathlet_dict.json', 'r') as f:
    pathlet_dict = json.load(f)

def is_consecutive_subsequence(sub, main_list):
    # Function to check if 'sub' is a consecutive subsequence of 'main_list'
    sub_len = len(sub)
    for i in range(len(main_list) - sub_len + 1):
        if main_list[i:i+sub_len] == sub:
            return True
    return False

class PathletEnvironment(gym.Env):
    def __init__(self, traj_edge_dict, pathlet_dict, M=0.25, mu_threshold=0.8, k=10):
        super(PathletEnvironment, self).__init__()

        self.traj_edge_dict = traj_edge_dict
        self.pathlet_dict = pathlet_dict
        self.max_weight = 0

        self.traj_edge_combined_dict = dict()
        for key, value in traj_edge_dict.items():
            combined_list = [node for edge in value for node in edge[:-1]] + [value[-1][-1]]
            self.traj_edge_combined_dict[key] = combined_list

        self.trajectory_coverage_pathlet = dict()
        self.trajectory_coverage = dict()
        self.initial_trajectory_coverage = dict()
        self.trajectory_weighted_coverage = dict()
        self.MAX_ACTION = 0
        self.MAX_pathlet_size = 0
        self.MAX_phi = 0
        self.step_count = 0

        self.M = M
        self.mu_threshold = mu_threshold
        self.k = k

        self.action_space = spaces.Discrete(10)  # Dummy value, will be updated in reset()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.pathlet_graph = self.initialize_pathlet_graph(self.pathlet_dict)
        self.MAX_pathlet_size = len(self.pathlet_graph)
        self.step_count = 0
        self.MAX_ACTION = max(len(val['neighbors']) for val in self.pathlet_graph.values())
        print("MAX_ACTION", self.MAX_ACTION)
        self.trajectories = list(self.traj_edge_dict.keys())
        for traj_id in self.trajectories:
            pathlet_lengths = 0
            self.trajectory_coverage_pathlet[traj_id] = list()
            for pathlet_id, traversal_set in self.pathlet_graph.items():
                if traj_id in traversal_set["traversal_set"]:
                    pathlet_lengths += self.pathlet_graph[pathlet_id]['length']
                    self.trajectory_coverage_pathlet[traj_id].append(pathlet_id)
            self.initial_trajectory_coverage[traj_id] = pathlet_lengths
            self.trajectory_coverage[traj_id] = pathlet_lengths / self.initial_trajectory_coverage[traj_id]
        self.unprocessed_pathlets = set(self.pathlet_graph.keys())
        self.candidate_pathlet_set = set()
        self.update_metrics()
        self.trajectory_loss = 0
        self.utility = 0
        self.init_phi = np.mean([len(pathlets) for traj, pathlets in self.trajectory_coverage_pathlet.items() if traj in self.trajectories])
        self.init_mu = np.mean([coverage for traj, coverage in self.trajectory_coverage.items() if traj in self.trajectories])
        self.init_size = len(self.pathlet_graph)
        self.current_pathlet = random.sample(list(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set)), k=1)[0]
        self.state = self.get_state()
        self.current_size = len(self.pathlet_graph)
        self.current_mu = np.mean([coverage for traj, coverage in self.trajectory_coverage.items() if traj in self.trajectories])
        self.current_phi = np.mean([len(pathlets) for traj, pathlets in self.trajectory_coverage_pathlet.items() if traj in self.trajectories])
        self.MAX_phi = max([len(pathlets) for traj, pathlets in self.trajectory_coverage_pathlet.items() if traj in self.trajectories])
        self.current_trajectory_loss = 0
        self.action_space = spaces.Discrete(self.MAX_ACTION + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5 + 1 * self.MAX_ACTION,), dtype=np.float32)
        return self.state, {}

    def initialize_pathlet_graph(self, pathlet_dict):
        pathlet_graph = {}
        self.max_weight = 0
        for pathlet_id, nodes in pathlet_dict.items():
            pathlet_graph[pathlet_id] = {
                'nodes': nodes,
                'length': 1,
                'traversal_set': set(),
                'neighbors': set(),
                'weight': 0
            }
            for traj_id, pathlets in traj_edge_dict.items():
                if is_consecutive_subsequence(pathlet_graph[pathlet_id]['nodes'], self.traj_edge_combined_dict[traj_id]) or is_consecutive_subsequence(pathlet_graph[pathlet_id]['nodes'][::-1], self.traj_edge_combined_dict[traj_id]):
                    pathlet_graph[pathlet_id]['traversal_set'].add(traj_id)
            pathlet_graph[pathlet_id]['weight'] = len(pathlet_graph[pathlet_id]['traversal_set']) / len(self.traj_edge_dict)
            self.max_weight = max(self.max_weight, pathlet_graph[pathlet_id]['weight'])
            head = pathlet_graph[pathlet_id]['nodes'][0]
            tail = pathlet_graph[pathlet_id]['nodes'][-1]
            for p_id, nodes in pathlet_graph.items():
                other_head = nodes['nodes'][0]
                other_tail = nodes['nodes'][-1]
                if p_id != pathlet_id and (head == other_tail or tail == other_head or head == other_head or tail == other_tail):
                    if len(pathlet_graph[pathlet_id]['nodes'] + pathlet_graph[p_id]['nodes']) == len(set(pathlet_graph[pathlet_id]['nodes'] + pathlet_graph[p_id]['nodes'])) + 1:
                        pathlet_graph[pathlet_id]['neighbors'].add(p_id)
        for pathlet_id, nodes in pathlet_graph.items():
            nodes['weight'] /= self.max_weight
        return pathlet_graph

    def get_state(self):
        max_neighbors = self.MAX_ACTION  # Define the maximum number of neighbors to consider
        current_pathlet_data = self.pathlet_graph[self.current_pathlet]
        neighbors = list(current_pathlet_data['neighbors'])
        
        state = []

        # normalized global information
        state.append(len(self.pathlet_graph) / self.MAX_pathlet_size)
        state.append(self.phi / self.MAX_phi)
        state.append(self.trajectory_loss)
        state.append(self.mu)
        #add current pathlet weight
        state.append(current_pathlet_data['weight'])
        # Neighbor pathlet information
        for i in range(max_neighbors):
            if i < len(neighbors):
                neighbor_data = self.pathlet_graph[neighbors[i]]
                state.append(neighbor_data['weight'])
            else:
                # Padding with zeros if there are fewer neighbors
                state.extend([0])
        
        state = np.array(state)
        return state


    def __change_init_action(self):
        return random.randint(1, self.MAX_ACTION)

    def step(self, action):
        reward = 0
        prev_size = self.current_size
        prev_mu = self.current_mu
        prev_phi = self.current_phi
        prev_trajectory_loss = self.current_trajectory_loss
        terminated = False
        if self.current_pathlet is None:
            print("current_pathlet is None")
            exit()
        if action == 0 and self.step_count == 0:
            action = self.__change_init_action()
        if action == 0 and self.step_count > 0:
            self.candidate_pathlet_set.add(self.current_pathlet)
            available_pathlets = list(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set))
            if available_pathlets:
                self.current_pathlet = random.sample(available_pathlets, k=1)[0]
            else:
                terminated = True
        else:
            neighbors = list(self.pathlet_graph[self.current_pathlet]['neighbors'])
            if len(neighbors) == 0:
                self.candidate_pathlet_set.add(self.current_pathlet)
                available_pathlets = list(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set))
                if not available_pathlets:
                    terminated = True
                else:
                    self.current_pathlet = random.sample(available_pathlets, k=1)[0]
                    
            else:
                neighbor_idx = self.get_action_neigh_ix_mapping(self.pathlet_graph[self.current_pathlet]['neighbors'], action)
                neighbor = list(self.pathlet_graph[self.current_pathlet]['neighbors'])[neighbor_idx]
                new_pathlet_id = self.merge_pathlets(self.current_pathlet, neighbor)
                self.remove_lost_trajectories()
                available_pathlets = list(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set))
                self.current_pathlet = new_pathlet_id if new_pathlet_id is not None else (random.sample(available_pathlets, k=1)[0] if available_pathlets else None)

        self.step_count += 1
        self.update_metrics()
        self.current_size = len(self.pathlet_graph)
        self.current_mu = self.mu
        self.current_phi = self.phi
        self.current_trajectory_loss = self.trajectory_loss
        self.state = self.get_state()
        terminated = self.check_done() or terminated
        truncated = False
        reward = self.calculate_reward(prev_size, prev_mu, prev_phi, prev_trajectory_loss, terminated)
        #if self.step_count % 100 == 0:
            #print("state", self.state[:4])
        if terminated:
            print("terminated", self.state)
            with open("pathlet_rl_plus_plus/obs.txt", "a") as file:
                file.write(f"state: {self.state[:4]}\n")
        return self.state, reward, terminated, truncated, {}




    def get_action_neigh_ix_mapping(self, neighbors, action):
        num_of_neighbors = len(neighbors)
        max_action = self.MAX_ACTION
        interval = max_action / num_of_neighbors

        lb = [i * interval for i in range(num_of_neighbors)]
        ub = [(i + 1) * interval for i in range(num_of_neighbors)]
        bin_bools = [lb[i] < action <= ub[i] for i in range(num_of_neighbors)]

        indices = np.where(bin_bools)[0]
        if len(indices) == 0:
            raise ValueError(f"Action {action} is out of bounds for the given neighbors.")
        
        return indices[0]

    def merge_pathlets(self, pathlet, neighbor):
        try:
            if self.pathlet_graph[pathlet]['length'] + self.pathlet_graph[neighbor]['length'] > self.k:
                self.candidate_pathlet_set.add(pathlet)
                self.current_pathlet = random.sample(list(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set)), k=1)[0]
                return None

            nodes_pathlet = set(self.pathlet_graph[pathlet]['nodes'])
            nodes_neighbor = set(self.pathlet_graph[neighbor]['nodes'])
            shared_node = nodes_pathlet.intersection(nodes_neighbor)
            
            if not shared_node:
                return None
            
            shared_node = shared_node.pop()

            nodes_pathlet = self.pathlet_graph[pathlet]['nodes']
            nodes_neighbor = self.pathlet_graph[neighbor]['nodes']
            
            if nodes_pathlet[-1] == shared_node and nodes_neighbor[0] == shared_node:
                merged_nodes = nodes_pathlet + nodes_neighbor[1:]
            elif nodes_pathlet[0] == shared_node and nodes_neighbor[-1] == shared_node:
                merged_nodes = nodes_neighbor + nodes_pathlet[1:]
            elif nodes_pathlet[-1] == shared_node and nodes_neighbor[-1] == shared_node:
                merged_nodes = nodes_pathlet + list(reversed(nodes_neighbor[:-1]))
            elif nodes_pathlet[0] == shared_node and nodes_neighbor[0] == shared_node:
                merged_nodes = list(reversed(nodes_pathlet[1:])) + nodes_neighbor
            else:
                return None
            
            new_pathlet_id = f"{pathlet}_{neighbor}"
            self.pathlet_graph[new_pathlet_id] = {
                'nodes': merged_nodes,
                'length': self.pathlet_graph[pathlet]['length'] + self.pathlet_graph[neighbor]['length'],
                'traversal_set': set(),
                'neighbors': set(),
                'weight': 0
            }
            for traj_id, pathlets in self.traj_edge_dict.items():
                if is_consecutive_subsequence(self.pathlet_graph[new_pathlet_id]['nodes'], self.traj_edge_combined_dict[traj_id]) or is_consecutive_subsequence(self.pathlet_graph[new_pathlet_id]['nodes'][::-1], self.traj_edge_combined_dict[traj_id]):
                    self.pathlet_graph[new_pathlet_id]['traversal_set'].add(traj_id)
            self.pathlet_graph[new_pathlet_id]['weight'] = len(self.pathlet_graph[new_pathlet_id]['traversal_set']) / len(self.traj_edge_dict)
            self.pathlet_graph[new_pathlet_id]['weight'] /= self.max_weight
            
            for p_id, nodes in self.pathlet_graph.items():
                if pathlet in nodes['neighbors']:
                    self.pathlet_graph[p_id]['neighbors'].remove(pathlet)
                if neighbor in nodes['neighbors']:
                    self.pathlet_graph[p_id]['neighbors'].remove(neighbor)
            del self.pathlet_graph[pathlet]
            del self.pathlet_graph[neighbor]

            head = self.pathlet_graph[new_pathlet_id]['nodes'][0]
            tail = self.pathlet_graph[new_pathlet_id]['nodes'][-1]
            
            for p_id, nodes in self.pathlet_graph.items():
                if p_id != new_pathlet_id and (head == nodes['nodes'][0] or head == nodes['nodes'][-1] or tail == nodes['nodes'][0] or tail == nodes['nodes'][-1]):
                    if len(self.pathlet_graph[new_pathlet_id]['nodes'] + self.pathlet_graph[p_id]['nodes']) == len(set(self.pathlet_graph[new_pathlet_id]['nodes'] + self.pathlet_graph[p_id]['nodes'])) + 1:
                        self.pathlet_graph[new_pathlet_id]['neighbors'].add(p_id)
                        self.pathlet_graph[p_id]['neighbors'].add(new_pathlet_id)
            
            for traj_id in self.trajectories:
                pathlet_lengths = 0
                self.trajectory_coverage_pathlet[traj_id] = list()
                for pathlet_id, traversal_set in self.pathlet_graph.items():
                    if traj_id in traversal_set["traversal_set"]:
                        pathlet_lengths += self.pathlet_graph[pathlet_id]['length']
                        self.trajectory_coverage_pathlet[traj_id].append(pathlet_id)
                self.trajectory_coverage[traj_id] = pathlet_lengths / self.initial_trajectory_coverage[traj_id]
            
            return new_pathlet_id
    
        except KeyError:
            raise

    def remove_lost_trajectories(self):
        lost_trajectories = [traj for traj, coverage in self.trajectory_coverage.items() if coverage == 0]
        self.trajectories = [traj for traj in self.trajectories if traj not in lost_trajectories]
        self.trajectory_loss = len(lost_trajectories) / len(self.traj_edge_dict)

    def update_metrics(self):
        self.phi = np.mean([len(pathlets) for traj, pathlets in self.trajectory_coverage_pathlet.items() if traj in self.trajectories])
        self.mu = np.mean([coverage for traj, coverage in self.trajectory_coverage.items() if traj in self.trajectories])

    def calculate_reward(self, prev_size, prev_mu, prev_phi, prev_trajectory_loss, done):
        # Constants for weighting different reward components
        alpha1, alpha2, alpha3, alpha4 = 0.25, 0.25, 0.25, 0.25

        # Calculate changes in metrics
        delta_size = (prev_size - self.current_size) / prev_size
        delta_phi = (prev_phi - self.current_phi) / prev_phi
        delta_traj_loss = (prev_trajectory_loss - self.current_trajectory_loss) / max(1, prev_size) if prev_trajectory_loss != 0 else 0
        delta_mu = (self.current_mu - prev_mu) / prev_mu
        
        # Calculate the distance from thresholds
        traj_loss_penalty = max(0.01, self.M - self.current_trajectory_loss) / self.M
        mu_penalty = max(0.01, self.current_mu - self.mu_threshold) / 0.2
        
        # The closer we are to the thresholds, the more penalty we apply
        traj_loss_weight = 1/traj_loss_penalty
        mu_weight = 1/mu_penalty
        
        # Calculate the reward considering penalties for being close to thresholds
        reward = (alpha1 * delta_size + 
                alpha2 * delta_phi + 
                alpha3 * delta_traj_loss * traj_loss_weight + 
                alpha4 * delta_mu * mu_weight)
        
        # Additional reward or penalty at the end of an episode
        if done:
            S1 = (self.init_size - self.current_size) / self.init_size
            S2 = (self.init_phi - self.current_phi) / self.init_phi
            S3 = 1 - self.current_trajectory_loss
            S4 = self.current_mu
            
            # Reward/penalty based on how well we did overall
            extra_reward = (alpha1 * S1 + alpha2 * S2 + alpha3 * S3 - alpha4 * S4)
            reward += extra_reward
        
        return reward




    def check_done(self):
        if self.trajectory_loss > self.M or self.mu < self.mu_threshold or len(set(self.pathlet_graph.keys()).difference(self.candidate_pathlet_set)) == 0:
            return True
        return False

    def render(self, mode='human'):
        pass

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class StateRewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int = 100, verbose: int = 1):
        super(StateRewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_states = []
        self.current_episode_rewards = []

    def _on_step(self) -> bool:
        # Collect rewards for the current episode
        reward = self.locals["rewards"]
        self.current_episode_rewards.append(reward)

        # Check if the episode has ended
        if self.locals["dones"]:
            # Calculate mean reward for the last 100 steps
            mean_reward = np.mean(self.current_episode_rewards[-self.check_freq:])
            last_state = self.locals["new_obs"]
            if self.verbose > 0:
                print(f"Episode Ended - Step: {self.num_timesteps}, Mean Reward (last {self.check_freq} steps): {mean_reward}")
                #save the text above
                with open("pathlet_rl_plus_plus/state_reward_logger.txt", "a") as file:
                    file.write(f"Episode Ended - Step: {self.num_timesteps}, Mean Reward (last {self.check_freq} steps): {mean_reward}\n")

            # Reset current episode rewards
            self.episode_rewards.append(self.current_episode_rewards)
            self.episode_states.append(last_state)
            self.current_episode_rewards = []

        return True

    def _on_rollout_end(self) -> None:
        # Called at the end of a rollout
        pass

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save the model
            model_path = f"{self.save_path}/model_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        return True

# Initialize the environment with arguments
env = make_vec_env(lambda: PathletEnvironment(traj_edge_dict, pathlet_dict, M=args.M, mu_threshold=args.mu_threshold, k=args.k), n_envs=1)

# Define policy architecture and create DQN agent with arguments
policy_kwargs = dict(
    net_arch=[128, 64, 32],
    activation_fn=torch.nn.ReLU
)
model = DQN(
    'MlpPolicy', 
    env, 
    learning_rate=args.lr, 
    buffer_size=args.buffer_size, 
    learning_starts=args.learning_starts,
    exploration_initial_eps=args.exploration_initial_eps,
    exploration_final_eps=args.exploration_final_eps,
    exploration_fraction=args.exploration_fraction,
    batch_size=args.batch_size, 
    gamma=args.gamma, 
    policy_kwargs=policy_kwargs, 
    verbose=1,
    seed=args.seed,
    device='cuda'
)


# Custom reward logger callback
state_reward_logger = StateRewardLoggerCallback(check_freq=100, verbose=1)
save_model_callback = SaveModelCallback(save_freq=1000, save_path='./pathlet_rl_plus_plus', verbose=1)
from os import makedirs
from os.path import exists
#create the folder
if not exists('./pathlet_rl_plus_plus'):
    makedirs('./pathlet_rl_plus_plus')

# Train the agent
model.learn(total_timesteps=args.timesteps, callback=[state_reward_logger, save_model_callback])
