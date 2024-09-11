import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class MultiArmedBanditEnv(gym.Env):
    def __init__(self, n_arms=10):
        super(MultiArmedBanditEnv, self).__init__()
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)  # no real observation
        self.expected_rewards = np.random.normal(0, 1, self.n_arms)

    def reset(self):
        return 0

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        reward = np.random.normal(
            self.expected_rewards[action], 1
        )  # reward is normally distributed
        done = True
        return 0, reward, done, False, {}

    def render(self, mode="human"):
        print(f"Arm probabilities: {self.expected_rewards}")

    def close(self):
        pass


class EpsilonGreedyGambler:
    def __init__(self, n=10, epsilon=0.1, alpha_fn=lambda n: 1 / n):
        super().__init__()
        self.n, self.epsilon, self.alpha_fn = n, epsilon, alpha_fn
       
        self.np_random = np.random.RandomState()

        self.Q = np.zeros(n)  # action values
        self.updates = np.zeros(n)  # number of updates for each arm
    
    def arm(self):
        if self.np_random.rand() < self.epsilon:  # exploration
            arm = self.np_random.randint(0, self.n)
        else:
            arm = np.argmax(self.Q)
        return arm

    def update(self, arm, reward):
        self.updates[arm] += 1
        self.Q[arm] += self.alpha_fn(self.updates[arm]) * (reward - self.Q[arm])


def run_single_bandit(epsilon, n_steps=1000, n_arms=10):
    """
    Simulates a single bandit problem over n_steps with a given epsilon value.
    """
    env = MultiArmedBanditEnv(n_arms=n_arms)
    gambler = EpsilonGreedyGambler(n=n_arms, epsilon=epsilon)

    optimal_arm = np.argmax(
        env.expected_rewards
    )  # optimal arm is the one with the highest expected reward
    rewards = np.zeros(n_steps)
    optimal_action_count = np.zeros(n_steps)

    for step in range(n_steps):
        arm = gambler.arm()
        _, reward, _, _, _ = env.step(arm)
        gambler.update(arm, reward)
        rewards[step] = reward  # record reward for this step
        if arm == optimal_arm:
            optimal_action_count[
                step
            ] = 1  # 1 if the optimal action was chosen, 0 otherwise

    return rewards, optimal_action_count


def run_multiple_bandits(epsilon, n_steps=1000, n_runs=2000, n_arms=10):
    """
    Simulates n_runs bandit problems and averages the rewards at each time step.
    Also tracks the percentage of optimal actions chosen.
    """
    all_rewards = np.zeros(n_steps)
    optimal_action_counts = np.zeros(n_steps)

    for run in range(n_runs):
        rewards, optimal_action_count = run_single_bandit(
            epsilon, n_steps=n_steps, n_arms=n_arms
        )
        all_rewards += rewards
        optimal_action_counts += optimal_action_count

    avg_rewards = all_rewards / n_runs
    optimal_action_percentage = (optimal_action_counts / n_runs) * 100
    return avg_rewards, optimal_action_percentage


def plot_results(n_runs=2000, n_steps=1000, n_arms=10):
    epsilons = [0, 0.01, 0.1]

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for epsilon in epsilons:
        avg_rewards, _ = run_multiple_bandits(
            epsilon, n_steps=n_steps, n_runs=n_runs, n_arms=n_arms
        )
        plt.plot(range(n_steps), avg_rewards, label=f"epsilon = {epsilon}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward over {n_runs} Runs for Different Epsilon Values")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for epsilon in epsilons:
        _, optimal_action_percentage = run_multiple_bandits(
            epsilon, n_steps=n_steps, n_runs=n_runs, n_arms=n_arms
        )
        plt.plot(
            range(n_steps), optimal_action_percentage, label=f"epsilon = {epsilon}"
        )
    plt.xlabel("Steps")
    plt.ylabel("Optimal Action (%)")
    plt.title(
        f"Percentage of Optimal Actions Chosen over {n_runs} Runs for Different Epsilon Values"
    )
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
