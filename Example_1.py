# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Save figures without opening GUI windows
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# 0. GLOBAL CONFIGURATION
# =========================================================
RESULT_ROOT = Path("simulation_results")
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Make simulation results more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x: Sequence[float], window: int = 20) -> np.ndarray:
    """Compute moving average for smoother plots."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. DEEP Q-NETWORK
# =========================================================
class DQN(nn.Module):
    def __init__(self, state_size: int, num_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# =========================================================
# 2. DYNAMIC SPECTRUM ENVIRONMENT
# =========================================================
class DynamicSpectrumEnv:
    """
    Partially observable dynamic spectrum environment.

    State used by the agent:
        h_t = finite history of action-observation pairs.

    True environment state:
        busy/free status of all channels, which is not directly observed
        by the agent.
    """

    def __init__(
        self,
        num_channels: int = 16,
        history_length: int = 4,
        busy_probabilities: Optional[Sequence[float]] = None,
    ):
        self.num_channels = num_channels
        self.history_length = history_length
        self.busy_probabilities = np.asarray(
            busy_probabilities if busy_probabilities is not None else [0.9] * 8 + [0.1] * 8,
            dtype=float,
        )

        if len(self.busy_probabilities) != self.num_channels:
            raise ValueError("busy_probabilities must have length equal to num_channels.")

        self.state = np.zeros(self.history_length * 2, dtype=float)

    def reset(self) -> np.ndarray:
        """Reset local observation history at the beginning of an episode."""
        self.state = np.zeros(self.history_length * 2, dtype=float)
        return self.state.copy()

    def set_busy_probabilities(self, busy_probabilities: Sequence[float]) -> None:
        busy_probabilities = np.asarray(busy_probabilities, dtype=float)
        if len(busy_probabilities) != self.num_channels:
            raise ValueError("busy_probabilities must have length equal to num_channels.")
        self.busy_probabilities = busy_probabilities

    def step(self, action: int) -> Tuple[np.ndarray, float, float]:
        """
        Execute one channel selection action.

        Returns:
            next_state: updated action-observation history
            reward: +1 if success, -1 if collision
            observation: 1 for ACK/success, 0 for NACK/collision
        """
        channel_busy = np.random.rand(self.num_channels) < self.busy_probabilities

        if not channel_busy[action]:
            reward = 1.0
            observation = 1.0
        else:
            reward = -1.0
            observation = 0.0

        next_state = np.roll(self.state, -2)
        next_state[-2] = action / (self.num_channels - 1)  # normalized action index
        next_state[-1] = observation
        self.state = next_state

        return next_state.copy(), reward, observation


# =========================================================
# 3. DQN AGENT WITH EXPERIENCE REPLAY AND TARGET NETWORK
# =========================================================
class DQLAgent:
    def __init__(
        self,
        state_size: int,
        num_channels: int,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        learning_rate: float = 0.001,
        memory_size: int = 2000,
    ):
        self.num_channels = num_channels
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)

        self.q_network = DQN(state_size, num_channels)
        self.target_network = DQN(state_size, num_channels)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.num_channels)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.memory.append((state.copy(), action, reward, next_state.copy()))

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())


# =========================================================
# 4. SIMULATION CONFIGURATION AND TRAINING FUNCTIONS
# =========================================================
@dataclass
class TrainConfig:
    num_channels: int = 16
    history_length: int = 4
    episodes: int = 500
    steps_per_episode: int = 100
    batch_size: int = 32
    target_update_freq: int = 10
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.998
    learning_rate: float = 0.001
    memory_size: int = 2000
    seed: int = DEFAULT_SEED


def train_dqn(
    config: TrainConfig,
    busy_probabilities: Sequence[float],
    scenario_dir: Path,
    switch_episode: Optional[int] = None,
    busy_probabilities_after_switch: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Tuple[DQLAgent, Dict[str, List[float]]]:
    """Train DQN agent and collect metrics."""
    ensure_dir(scenario_dir)
    set_seed(config.seed)

    env = DynamicSpectrumEnv(
        num_channels=config.num_channels,
        history_length=config.history_length,
        busy_probabilities=busy_probabilities,
    )

    agent = DQLAgent(
        state_size=config.history_length * 2,
        num_channels=config.num_channels,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        learning_rate=config.learning_rate,
        memory_size=config.memory_size,
    )

    metrics: Dict[str, List[float]] = {
        "episode": [],
        "reward": [],
        "success_rate": [],
        "collision_rate": [],
        "epsilon": [],
    }
    action_counts = np.zeros(config.num_channels, dtype=int)

    if verbose:
        print(f"\n--- Training scenario: {scenario_dir.name} ---")

    for episode in range(config.episodes):
        if switch_episode is not None and episode == switch_episode:
            if busy_probabilities_after_switch is None:
                raise ValueError("busy_probabilities_after_switch is required when switch_episode is used.")
            env.set_busy_probabilities(busy_probabilities_after_switch)

        state = env.reset()
        total_reward = 0.0
        success_count = 0

        for _ in range(config.steps_per_episode):
            action = agent.act(state, training=True)
            next_state, reward, observation = env.step(action)

            agent.remember(state, action, reward, next_state)
            agent.replay(config.batch_size)

            state = next_state
            total_reward += reward
            success_count += int(observation == 1.0)
            action_counts[action] += 1

        if episode % config.target_update_freq == 0:
            agent.update_target_network()

        success_rate = success_count / config.steps_per_episode
        collision_rate = 1.0 - success_rate

        metrics["episode"].append(episode + 1)
        metrics["reward"].append(total_reward)
        metrics["success_rate"].append(success_rate)
        metrics["collision_rate"].append(collision_rate)
        metrics["epsilon"].append(agent.epsilon)

        if verbose and (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1:4d}/{config.episodes}, "
                f"Reward = {total_reward:6.1f}, "
                f"Success = {success_rate:5.2%}, "
                f"Epsilon = {agent.epsilon:.3f}"
            )

    # Save action counts and metrics
    metrics["action_counts"] = action_counts.tolist()
    save_metrics_csv(metrics, scenario_dir / "metrics.csv")
    save_summary_txt(metrics, scenario_dir / "summary.txt")

    return agent, metrics


def evaluate_random_policy(
    config: TrainConfig,
    busy_probabilities: Sequence[float],
    num_eval_episodes: int = 200,
) -> Dict[str, float]:
    """Evaluate random channel selection baseline."""
    set_seed(config.seed + 999)
    env = DynamicSpectrumEnv(
        num_channels=config.num_channels,
        history_length=config.history_length,
        busy_probabilities=busy_probabilities,
    )

    rewards = []
    success_rates = []
    action_counts = np.zeros(config.num_channels, dtype=int)

    for _ in range(num_eval_episodes):
        env.reset()
        total_reward = 0.0
        success_count = 0

        for _ in range(config.steps_per_episode):
            action = random.randrange(config.num_channels)
            _, reward, observation = env.step(action)
            total_reward += reward
            success_count += int(observation == 1.0)
            action_counts[action] += 1

        rewards.append(total_reward)
        success_rates.append(success_count / config.steps_per_episode)

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_success_rate": float(np.mean(success_rates)),
        "avg_collision_rate": float(1.0 - np.mean(success_rates)),
        "action_counts": action_counts.tolist(),
    }


# =========================================================
# 5. SAVE RESULTS
# =========================================================
def save_metrics_csv(metrics: Dict[str, List[float]], path: Path) -> None:
    keys = ["episode", "reward", "success_rate", "collision_rate", "epsilon"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(metrics["episode"])):
            writer.writerow([metrics[k][i] for k in keys])


def save_summary_txt(metrics: Dict[str, List[float]], path: Path) -> None:
    last_n = min(100, len(metrics["reward"]))
    avg_reward = np.mean(metrics["reward"][-last_n:])
    avg_success = np.mean(metrics["success_rate"][-last_n:])
    avg_collision = np.mean(metrics["collision_rate"][-last_n:])

    with path.open("w", encoding="utf-8") as f:
        f.write("Summary over the last episodes\n")
        f.write(f"Number of episodes used: {last_n}\n")
        f.write(f"Average reward: {avg_reward:.4f}\n")
        f.write(f"Average success rate: {avg_success:.4f}\n")
        f.write(f"Average collision rate: {avg_collision:.4f}\n")


def plot_training_curves(metrics: Dict[str, List[float]], scenario_dir: Path, title_prefix: str = "DQN") -> None:
    episodes = metrics["episode"]

    # Reward
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, metrics["reward"], label="Reward per episode")
    ma = moving_average(metrics["reward"], window=20)
    if len(ma) < len(episodes):
        plt.plot(episodes[len(episodes) - len(ma):], ma, label="Moving average")
    plt.title(f"{title_prefix}: Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "01_reward_curve.png", dpi=300)
    plt.close()

    # Epsilon
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, metrics["epsilon"], label="Epsilon")
    plt.title(f"{title_prefix}: Exploration Rate")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "02_epsilon_curve.png", dpi=300)
    plt.close()

    # Success rate
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, metrics["success_rate"], label="Success rate")
    ma = moving_average(metrics["success_rate"], window=20)
    if len(ma) < len(episodes):
        plt.plot(episodes[len(episodes) - len(ma):], ma, label="Moving average")
    plt.title(f"{title_prefix}: Transmission Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "03_success_rate_curve.png", dpi=300)
    plt.close()

    # Collision rate
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, metrics["collision_rate"], label="Collision rate")
    ma = moving_average(metrics["collision_rate"], window=20)
    if len(ma) < len(episodes):
        plt.plot(episodes[len(episodes) - len(ma):], ma, label="Moving average")
    plt.title(f"{title_prefix}: Collision Rate")
    plt.xlabel("Episode")
    plt.ylabel("Collision rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "04_collision_rate_curve.png", dpi=300)
    plt.close()


def plot_action_distribution(action_counts: Sequence[int], scenario_dir: Path, title: str) -> None:
    channels = np.arange(1, len(action_counts) + 1)
    plt.figure(figsize=(8, 4))
    plt.bar(channels, action_counts)
    plt.title(title)
    plt.xlabel("Channel index")
    plt.ylabel("Number of selections")
    plt.xticks(channels)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(scenario_dir / "05_channel_selection_distribution.png", dpi=300)
    plt.close()


def plot_dqn_vs_random(
    dqn_metrics: Dict[str, List[float]],
    random_metrics: Dict[str, float],
    scenario_dir: Path,
) -> None:
    ensure_dir(scenario_dir)

    last_n = min(100, len(dqn_metrics["reward"]))
    dqn_avg_reward = float(np.mean(dqn_metrics["reward"][-last_n:]))
    dqn_success = float(np.mean(dqn_metrics["success_rate"][-last_n:]))
    dqn_collision = 1.0 - dqn_success

    labels = ["Random", "DQN"]
    avg_rewards = [random_metrics["avg_reward"], dqn_avg_reward]
    success_rates = [random_metrics["avg_success_rate"], dqn_success]
    collision_rates = [random_metrics["avg_collision_rate"], dqn_collision]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, avg_rewards)
    plt.title("Average Reward Comparison")
    plt.ylabel("Average reward per episode")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(scenario_dir / "01_average_reward_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, success_rates, width, label="Success rate")
    plt.bar(x + width / 2, collision_rates, width, label="Collision rate")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title("Success and Collision Rate Comparison")
    plt.ylabel("Rate")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "02_success_collision_comparison.png", dpi=300)
    plt.close()

    with (scenario_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Average Reward", "Success Rate", "Collision Rate"])
        writer.writerow(["Random", random_metrics["avg_reward"], random_metrics["avg_success_rate"], random_metrics["avg_collision_rate"]])
        writer.writerow(["DQN", dqn_avg_reward, dqn_success, dqn_collision])


def plot_history_length_comparison(results: Dict[int, Dict[str, List[float]]], scenario_dir: Path) -> None:
    ensure_dir(scenario_dir)

    plt.figure(figsize=(8, 5))
    for history_length, metrics in results.items():
        ma = moving_average(metrics["reward"], window=20)
        episodes = metrics["episode"]
        if len(ma) < len(episodes):
            x = episodes[len(episodes) - len(ma):]
        else:
            x = episodes
        plt.plot(x, ma, label=f"L = {history_length}")
    plt.title("Effect of History Length on Reward")
    plt.xlabel("Episode")
    plt.ylabel("Moving-average reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "01_history_length_reward_comparison.png", dpi=300)
    plt.close()

    labels = []
    success_values = []
    reward_values = []
    for history_length, metrics in results.items():
        last_n = min(100, len(metrics["reward"]))
        labels.append(f"L={history_length}")
        success_values.append(float(np.mean(metrics["success_rate"][-last_n:])))
        reward_values.append(float(np.mean(metrics["reward"][-last_n:])))

    plt.figure(figsize=(7, 4))
    plt.bar(labels, success_values)
    plt.title("Final Success Rate for Different History Lengths")
    plt.xlabel("History length")
    plt.ylabel("Average success rate")
    plt.ylim(0, 1)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(scenario_dir / "02_history_length_success_rate.png", dpi=300)
    plt.close()

    with (scenario_dir / "history_length_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["History length L", "Input size 2L", "Average Reward", "Success Rate"])
        for label, reward_value, success_value in zip(labels, reward_values, success_values):
            L = int(label.split("=")[1])
            writer.writerow([L, 2 * L, reward_value, success_value])


def plot_switch_marker(metrics: Dict[str, List[float]], scenario_dir: Path, switch_episode: int) -> None:
    episodes = metrics["episode"]

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, metrics["reward"], label="Reward per episode")
    plt.axvline(switch_episode, linestyle="--", label="Environment switch")
    ma = moving_average(metrics["reward"], window=20)
    if len(ma) < len(episodes):
        plt.plot(episodes[len(episodes) - len(ma):], ma, label="Moving average")
    plt.title("DQN Adaptation in a Time-Varying Environment")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "06_reward_with_environment_switch.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, metrics["success_rate"], label="Success rate")
    plt.axvline(switch_episode, linestyle="--", label="Environment switch")
    ma = moving_average(metrics["success_rate"], window=20)
    if len(ma) < len(episodes):
        plt.plot(episodes[len(episodes) - len(ma):], ma, label="Moving average")
    plt.title("Success Rate in a Time-Varying Environment")
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_dir / "07_success_rate_with_environment_switch.png", dpi=300)
    plt.close()


# =========================================================
# 6. SCENARIOS
# =========================================================
def scenario_1_basic(config: TrainConfig, root: Path) -> Dict[str, List[float]]:
    scenario_dir = root / "01_basic_environment"
    busy_prob = [0.9] * 8 + [0.1] * 8

    _, metrics = train_dqn(config, busy_prob, scenario_dir)
    plot_training_curves(metrics, scenario_dir, title_prefix="Basic Environment")
    plot_action_distribution(
        metrics["action_counts"],
        scenario_dir,
        title="Channel Selection Distribution after Training",
    )
    return metrics


def scenario_2_random_comparison(
    config: TrainConfig,
    root: Path,
    dqn_metrics: Optional[Dict[str, List[float]]] = None,
) -> None:
    scenario_dir = root / "02_dqn_vs_random"
    ensure_dir(scenario_dir)
    busy_prob = [0.9] * 8 + [0.1] * 8

    if dqn_metrics is None:
        _, dqn_metrics = train_dqn(config, busy_prob, scenario_dir / "dqn_training")

    random_metrics = evaluate_random_policy(config, busy_prob, num_eval_episodes=200)
    plot_dqn_vs_random(dqn_metrics, random_metrics, scenario_dir)
    plot_action_distribution(
        random_metrics["action_counts"],
        scenario_dir,
        title="Random Policy Channel Selection Distribution",
    )


def scenario_3_history_length(config: TrainConfig, root: Path) -> None:
    scenario_dir = root / "03_history_length_comparison"
    ensure_dir(scenario_dir)
    busy_prob = [0.9] * 8 + [0.1] * 8

    results: Dict[int, Dict[str, List[float]]] = {}
    for L in [1, 2, 4, 8]:
        sub_config = TrainConfig(
            num_channels=config.num_channels,
            history_length=L,
            episodes=config.episodes,
            steps_per_episode=config.steps_per_episode,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            epsilon_decay=config.epsilon_decay,
            learning_rate=config.learning_rate,
            memory_size=config.memory_size,
            seed=config.seed + L,
        )
        sub_dir = scenario_dir / f"L_{L}"
        _, metrics = train_dqn(sub_config, busy_prob, sub_dir, verbose=False)
        plot_training_curves(metrics, sub_dir, title_prefix=f"History Length L={L}")
        plot_action_distribution(
            metrics["action_counts"],
            sub_dir,
            title=f"Channel Selection Distribution, L={L}",
        )
        results[L] = metrics
        print(f"Finished history length L={L}")

    plot_history_length_comparison(results, scenario_dir)


def scenario_4_time_varying_environment(config: TrainConfig, root: Path) -> None:
    scenario_dir = root / "04_time_varying_environment"
    busy_prob_before = [0.9] * 8 + [0.1] * 8
    busy_prob_after = [0.1] * 8 + [0.9] * 8
    switch_episode = config.episodes // 2

    _, metrics = train_dqn(
        config,
        busy_prob_before,
        scenario_dir,
        switch_episode=switch_episode,
        busy_probabilities_after_switch=busy_prob_after,
    )

    plot_training_curves(metrics, scenario_dir, title_prefix="Time-Varying Environment")
    plot_action_distribution(
        metrics["action_counts"],
        scenario_dir,
        title="Overall Channel Selection Distribution in Time-Varying Environment",
    )
    plot_switch_marker(metrics, scenario_dir, switch_episode=switch_episode)


# =========================================================
# 7. MAIN
# =========================================================
def main() -> None:
    config = TrainConfig(
        num_channels=16,
        history_length=4,
        episodes=500,
        steps_per_episode=100,
        batch_size=32,
        target_update_freq=10,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        learning_rate=0.001,
        memory_size=2000,
        seed=DEFAULT_SEED,
    )

    root = RESULT_ROOT
    ensure_dir(root)

    print("Saving all simulation results to:", root.resolve())

    dqn_metrics = scenario_1_basic(config, root)
    scenario_2_random_comparison(config, root, dqn_metrics=dqn_metrics)
    scenario_3_history_length(config, root)
    scenario_4_time_varying_environment(config, root)

    print("\nDone. All figures and CSV files were saved successfully.")
    print("Output folder:", root.resolve())


if __name__ == "__main__":
    main()
