import argparse
import datetime
import json
import os
import time
from importlib import import_module
from typing import Any, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter


class JsonLogger:
    """Structured JSON logger that stores experiment events grouped in a single file."""

    def __init__(self, filepath: str, auto_flush: bool = True):
        self.filepath = filepath
        log_dir = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(log_dir, exist_ok=True)
        self._auto_flush = auto_flush
        self._records: list[Dict[str, Any]] = []

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as existing_file:
                    existing_data = json.load(existing_file)
                    if isinstance(existing_data, dict) and "events" in existing_data:
                        events = existing_data["events"]
                        if isinstance(events, list):
                            self._records.extend(events)
            except json.JSONDecodeError:
                # If the file is corrupted or not valid JSON, start fresh.
                self._records = []

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "event": event,
            **payload,
        }
        self._records.append(record)
        if self._auto_flush:
            self.flush()

    def flush(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as output_file:
            json.dump({"events": self._records}, output_file, indent=2)

    def close(self) -> None:
        if not self._auto_flush:
            self.flush()

    def __enter__(self) -> "JsonLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class RawRewardTracker(gym.Wrapper):
    """Injects unnormalized per-episode rewards into info dict regardless of external early stops."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_raw_return = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._episode_raw_return = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_raw_return += reward
        info = dict(info) if info else {}
        info["raw_episode_return"] = self._episode_raw_return
        info["raw_step_reward"] = reward
        if terminated or truncated:
            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                episode_info["raw_return"] = self._episode_raw_return
            self._episode_raw_return = 0.0
        return observation, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL algorithms with PyTorch in MuJoCo environments (step-based driver)"
    )
    parser.add_argument("--env", type=str, default="Humanoid-v2", help="Environment id")
    parser.add_argument(
        "--algo",
        type=str,
        default="atac",
        help="Algorithm: vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac",
    )
    parser.add_argument(
        "--phase", type=str, default="train", choices=["train", "test"], help="Phase"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--load", type=str, default=None, help="Checkpoint name to load")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_train_steps", type=int, default=1_000_000, help="Total training steps"
    )
    parser.add_argument(
        "--steps_per_iter",
        type=int,
        default=5_000,
        help="Environment steps collected per outer iter",
    )
    parser.add_argument("--max_step", type=int, default=1_000, help="Max steps per episode")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enable TensorBoard logging",
    )
    parser.add_argument("--gpu_index", type=int, default=0, help="CUDA device index")

    # Shared hyperparameters for on-policy agents
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.97, help="GAE lambda")

    # PPO hyperparameters
    parser.add_argument("--ppo_sample_size", type=int, default=2048, help="PPO batch size")
    parser.add_argument(
        "--ppo_train_policy_iters", type=int, default=10, help="Policy gradient steps"
    )
    parser.add_argument(
        "--ppo_train_vf_iters", type=int, default=10, help="Value function steps"
    )
    parser.add_argument("--ppo_minibatch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--ppo_clip_param", type=float, default=0.2, help="Clip epsilon")
    parser.add_argument("--ppo_target_kl", type=float, default=0.01, help="Target KL")
    parser.add_argument("--ppo_policy_lr", type=float, default=3e-4, help="Policy LR")
    parser.add_argument("--ppo_vf_lr", type=float, default=1e-3, help="Value function LR")

    # TRPO/NPG hyperparameters
    parser.add_argument(
        "--trpo_sample_size", type=int, default=2048, help="TRPO advantage batch size"
    )
    parser.add_argument("--trpo_vf_lr", type=float, default=1e-3, help="TRPO value LR")
    parser.add_argument(
        "--trpo_train_vf_iters", type=int, default=80, help="TRPO value iterations"
    )
    parser.add_argument(
        "--trpo_minibatch_size", type=int, default=64, help="TRPO value minibatch"
    )
    parser.add_argument("--trpo_delta", type=float, default=0.01, help="TRPO KL constraint")
    parser.add_argument(
        "--trpo_backtrack_iter", type=int, default=10, help="Backtracking iterations"
    )
    parser.add_argument(
        "--trpo_backtrack_coeff",
        type=float,
        default=1.0,
        help="Initial backtracking coefficient",
    )
    parser.add_argument(
        "--trpo_backtrack_alpha",
        type=float,
        default=0.5,
        help="Backtracking acceptance threshold",
    )

    # Evaluation and logging controls
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        choices=["tune", "compare"],
        default="compare",
        help="Evaluation scheduling mode",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Number of offline episodes per evaluation",
    )
    parser.add_argument(
        "--eval_rollout_steps",
        type=int,
        default=None,
        help="Max steps during evaluation rollouts (defaults to max_step)",
    )
    parser.add_argument(
        "--tune_eval_window_steps",
        type=int,
        default=100_000,
        help="Training steps reserved for evaluations in tune mode",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200_000,
        help="Steps between intermediate checkpoints",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=20000,
        help="Evaluation frequency in steps",
        )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to store JSONL experiment logs",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./save_model",
        help="Directory to store model checkpoints",
    )
    return parser.parse_args()


def get_agent_class(algo: str):
    algo_lower = algo.lower()
    if algo_lower == "vpg":
        module = import_module("agents.vpg")
    elif algo_lower in {"npg", "trpo"}:
        module = import_module("agents.trpo")
    elif algo_lower == "ppo":
        module = import_module("agents.ppo")
    elif algo_lower == "ddpg":
        module = import_module("agents.ddpg")
    elif algo_lower == "td3":
        module = import_module("agents.td3")
    elif algo_lower in {"sac", "asac", "tac", "atac"}:
        module = import_module("agents.sac")
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return module.Agent


def instantiate_agent(
    env,
    args: argparse.Namespace,
    device: torch.device,
    obs_dim: int,
    act_dim: int,
    act_limit: float,
) -> Tuple[Any, Dict[str, Any]]:
    algo_lower = args.algo.lower()
    AgentClass = get_agent_class(algo_lower)
    agent_kwargs: Dict[str, Any] = {}
    agent_hparams: Dict[str, Any] = {}

    if algo_lower in {"ddpg", "td3"}:
        agent_kwargs = {
            "expl_before": 10_000,
            "act_noise": 0.1,
            "hidden_sizes": (256, 256),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
        }
        agent_hparams.update(agent_kwargs)
    elif algo_lower == "sac":
        agent_kwargs = {
            "expl_before": 10_000,
            "alpha": 0.2,
            "hidden_sizes": (256, 256),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
        }
        agent_hparams.update(agent_kwargs)
    elif algo_lower == "asac":
        agent_kwargs = {
            "expl_before": 10_000,
            "automatic_entropy_tuning": True,
            "hidden_sizes": (256, 256),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
        }
        agent_hparams.update(agent_kwargs)
    elif algo_lower == "tac":
        agent_kwargs = {
            "expl_before": 10_000,
            "alpha": 0.2,
            "log_type": "log-q",
            "entropic_index": 1.2,
            "hidden_sizes": (256, 256),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
        }
        agent_hparams.update(agent_kwargs)
    elif algo_lower == "atac":
        agent_kwargs = {
            "expl_before": 10_000,
            "log_type": "log-q",
            "entropic_index": 1.2,
            "automatic_entropy_tuning": True,
            "hidden_sizes": (256, 256),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
        }
        agent_hparams.update(agent_kwargs)
    elif algo_lower == "ppo":
        agent_kwargs = {
            "sample_size": args.ppo_sample_size,
            "train_policy_iters": args.ppo_train_policy_iters,
            "train_vf_iters": args.ppo_train_vf_iters,
            "minibatch_size": args.ppo_minibatch_size,
            "clip_param": args.ppo_clip_param,
            "target_kl": args.ppo_target_kl,
            "policy_lr": args.ppo_policy_lr,
            "vf_lr": args.ppo_vf_lr,
            "gamma": args.gamma,
            "lam": args.lam,
        }
        agent_hparams.update(
            {
                "sample_size": args.ppo_sample_size,
                "train_policy_iters": args.ppo_train_policy_iters,
                "train_vf_iters": args.ppo_train_vf_iters,
                "minibatch_size": args.ppo_minibatch_size,
                "clip_param": args.ppo_clip_param,
                "target_kl": args.ppo_target_kl,
                "policy_lr": args.ppo_policy_lr,
                "vf_lr": args.ppo_vf_lr,
                "gamma": args.gamma,
                "lam": args.lam,
            }
        )
    elif algo_lower in {"trpo", "npg"}:
        agent_kwargs = {
            "sample_size": args.trpo_sample_size,
            "vf_lr": args.trpo_vf_lr,
            "train_vf_iters": args.trpo_train_vf_iters,
            "minibatch_size": args.trpo_minibatch_size,
            "delta": args.trpo_delta,
            "backtrack_iter": args.trpo_backtrack_iter,
            "backtrack_coeff": args.trpo_backtrack_coeff,
            "backtrack_alpha": args.trpo_backtrack_alpha,
            "gamma": args.gamma,
            "lam": args.lam,
        }
        agent_hparams.update(
            {
                "sample_size": args.trpo_sample_size,
                "vf_lr": args.trpo_vf_lr,
                "train_vf_iters": args.trpo_train_vf_iters,
                "minibatch_size": args.trpo_minibatch_size,
                "delta": args.trpo_delta,
                "backtrack_iter": args.trpo_backtrack_iter,
                "backtrack_coeff": args.trpo_backtrack_coeff,
                "backtrack_alpha": args.trpo_backtrack_alpha,
                "gamma": args.gamma,
                "lam": args.lam,
            }
        )
    else:
        agent_kwargs = {
            "sample_size": args.ppo_sample_size,
            "train_policy_iters": args.ppo_train_policy_iters,
            "train_vf_iters": args.ppo_train_vf_iters,
            "gamma": args.gamma,
            "lam": args.lam,
        }
        agent_hparams.update(
            {
                "sample_size": args.ppo_sample_size,
                "train_policy_iters": args.ppo_train_policy_iters,
                "train_vf_iters": args.ppo_train_vf_iters,
                "gamma": args.gamma,
                "lam": args.lam,
            }
        )

    agent = AgentClass(env, args, device, obs_dim, act_dim, act_limit, **agent_kwargs)
    return agent, agent_hparams


def format_for_name(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "x".join(format_for_name(v) for v in value)
    return str(value)


def build_experiment_name(params: Dict[str, Any]) -> str:
    priority_keys = ["env", "algo", "phase", "seed"]
    ordered_keys: Iterable[str] = [
        *[key for key in priority_keys if key in params],
        *sorted(key for key in params if key not in priority_keys),
    ]
    parts = []
    ppo_keys = [
        "sample_size",
        "clip_param",
        "target_kl",
        "policy_lr",
        "vf_lr",
        "train_policy_iters",
        "train_vf_iters",
        "gamma",
        "lam",
    ]
    trpo_keys = [
        "sample_size",
        "delta",
        "backtrack_iter",
        "backtrack_coeff",
        "backtrack_alpha",
        "train_vf_iters",
        "gamma",
        "lam",
    ]

    for key in ordered_keys:
        value = params[key]
        if key not in ppo_keys and key not in trpo_keys and key not in priority_keys:
            continue
        parts.append(f"{key}={format_for_name(value)}")
    name = "__".join(parts)
    return name.replace("/", "-")


def evaluate_agent(agent, num_episodes: int, max_step: int) -> Dict[str, Any]:
    agent.eval_mode = True
    episode_returns = []
    episode_lengths = []
    for _ in range(num_episodes):
        length, episode_return = agent.run(max_step)
        episode_lengths.append(length)
        episode_returns.append(episode_return)
    agent.eval_mode = False

    returns_array = np.array(episode_returns, dtype=np.float32)
    lengths_array = np.array(episode_lengths, dtype=np.int32)
    return {
        "num_episodes": num_episodes,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "average_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "std_return": float(returns_array.std()) if returns_array.size else 0.0,
        "min_return": float(returns_array.min()) if returns_array.size else 0.0,
        "max_return": float(returns_array.max()) if returns_array.size else 0.0,
        "average_length": float(lengths_array.mean()) if lengths_array.size else 0.0,
    }


def main():
    args = parse_args()
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)
    env = Monitor(env)
    env = RawRewardTracker(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    agent, agent_hparams = instantiate_agent(env, args, device, obs_dim, act_dim, act_limit)

    if args.load is not None:
        checkpoint_dir = os.path.abspath(args.save_model_dir)
        checkpoint_path = os.path.join(checkpoint_dir, str(args.load))
        pretrained_model = torch.load(checkpoint_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    eval_step_limit = args.eval_rollout_steps or args.max_step

    experiment_params = {
        "env": args.env,
        "algo": args.algo,
        "phase": args.phase,
        "seed": args.seed,
        "total_steps": args.total_train_steps,
        "steps_per_iter": args.steps_per_iter,
        "max_step": args.max_step,
        "eval_mode": args.evaluation_mode,
        "eval_episodes": args.eval_episodes,
        "eval_step_limit": eval_step_limit,
        "tune_eval_window_steps": args.tune_eval_window_steps,
        **agent_hparams,
    }
    experiment_name = build_experiment_name(experiment_params)

    log_dir = os.path.abspath(args.log_dir)
    log_path = os.path.join(log_dir, f"{experiment_name}.json")

    tensorboard_writer = None
    tensorboard_dir = None
    if args.tensorboard and args.load is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tensorboard_dir = os.path.join("runs", experiment_name, timestamp)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

    os.makedirs(os.path.abspath(args.save_model_dir), exist_ok=True)
    
    tune_eval_step_threshold = max(args.total_train_steps - args.tune_eval_window_steps, 0)

    total_num_steps = 0
    train_sum_returns = 0.0
    train_num_episodes = 0
    next_eval_step = args.eval_freq
    next_checkpoint_step = args.checkpoint_freq
    last_eval_metrics: Optional[Dict[str, Any]] = None

    start_time = time.time()

    with JsonLogger(log_path) as logger:
        logger.log(
            "experiment_start",
            {
                "experiment_name": experiment_name,
                "config": experiment_params,
                "log_path": log_path,
                "tensorboard_dir": tensorboard_dir,
            },
        )

        while total_num_steps < args.total_train_steps:
            if args.phase == "train":
                train_step_count = 0
            
                agent.eval_mode = False
                remaining_steps = args.total_train_steps - total_num_steps
                steps_for_episode = min(args.max_step, remaining_steps)
                episode_length, episode_return = agent.run(steps_for_episode)

                total_num_steps += episode_length
                train_step_count += episode_length
                train_num_episodes += 1
                train_sum_returns += episode_return

                train_average_return = (
                    train_sum_returns / train_num_episodes if train_num_episodes else 0.0
                )
                elapsed_time = time.time() - start_time

                logger.log(
                    "train_episode",
                    {
                        "total_steps": total_num_steps,
                        "episode_index": train_num_episodes,
                        "episode_length": episode_length,
                        "episode_return": episode_return,
                        "average_return": train_average_return,
                        "agent_metrics": dict(agent.logger),
                        "elapsed_time_sec": elapsed_time,
                    },
                )

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        "Train/AverageReturn", train_average_return, total_num_steps
                    )
                    tensorboard_writer.add_scalar(
                        "Train/EpisodeReturn", episode_return, total_num_steps
                    )
                    if args.algo.lower() in {"asac", "atac"} and hasattr(agent, "alpha"):
                        tensorboard_writer.add_scalar(
                            "Train/Alpha", getattr(agent, "alpha"), total_num_steps
                        )
                    if "BacktrackIter" in agent.logger:
                        tensorboard_writer.add_scalar("Train/BacktrackIter", agent.logger["BacktrackIter"], total_num_steps)

            should_evaluate = False
            if args.phase == "test":
                should_evaluate = True
            elif args.evaluation_mode == "compare" and  total_num_steps >= next_eval_step:
                should_evaluate = True
                next_eval_step += args.eval_freq
            elif (
                args.evaluation_mode == "tune"
                and total_num_steps >= tune_eval_step_threshold
                and total_num_steps >= next_eval_step
            ):
                should_evaluate = True
                next_eval_step += args.eval_freq

            if should_evaluate or total_num_steps >= args.total_train_steps:
                eval_metrics = evaluate_agent(agent, args.eval_episodes, eval_step_limit)
                eval_metrics.update(
                    {
                        "total_steps": total_num_steps,
                        "train_episodes_completed": train_num_episodes,
                        "mode": args.evaluation_mode,
                    }
                )
                last_eval_metrics = eval_metrics
                logger.log("evaluation", eval_metrics)

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        "Eval/AverageReturn", eval_metrics["average_return"], total_num_steps
                    )
                    tensorboard_writer.add_scalar(
                        "Eval/StdReturn", eval_metrics["std_return"], total_num_steps
                    )

            if args.phase == "test":
                break

            if args.phase == "train" and args.load is None:
                checkpoint_due = total_num_steps >= next_checkpoint_step
                within_tune_window = (
                    args.evaluation_mode != "tune"
                    or total_num_steps >= tune_eval_step_threshold
                )
                if checkpoint_due and within_tune_window:
                    if last_eval_metrics is None:
                        last_eval_metrics = evaluate_agent(
                            agent, args.eval_episodes, eval_step_limit
                        )
                    avg_return = last_eval_metrics["average_return"]
                    checkpoint_name = (
                        f"{experiment_name}__steps={total_num_steps}__avg={avg_return:.2f}.pt"
                    )
                    checkpoint_path = os.path.join(args.save_model_dir, checkpoint_name)
                    torch.save(agent.policy.state_dict(), checkpoint_path)
                    logger.log(
                        "checkpoint",
                        {
                            "checkpoint_path": checkpoint_path,
                            "total_steps": total_num_steps,
                            "average_return": avg_return,
                        },
                    )
                    next_checkpoint_step += args.checkpoint_freq
                if last_eval_metrics is None:
                    last_eval_metrics = evaluate_agent(agent, args.eval_episodes, eval_step_limit)
                avg_return = last_eval_metrics["average_return"]
                checkpoint_name = (
                    f"{experiment_name}__steps={total_num_steps}__avg={avg_return:.2f}.pt"
                )
                checkpoint_path = os.path.join(args.save_model_dir, checkpoint_name)
                torch.save(agent.policy.state_dict(), checkpoint_path)
                logger.log(
                    "checkpoint",
                    {
                        "checkpoint_path": checkpoint_path,
                        "total_steps": total_num_steps,
                        "average_return": avg_return,
                    },
                )
                next_checkpoint_step += args.checkpoint_freq

        if args.phase == "train" and last_eval_metrics is None:
            last_eval_metrics = evaluate_agent(agent, args.eval_episodes, eval_step_limit)
            last_eval_metrics.update(
                {
                    "total_steps": total_num_steps,
                    "train_episodes_completed": train_num_episodes,
                    "mode": args.evaluation_mode,
                }
            )
            logger.log("evaluation", last_eval_metrics)

        if args.phase == "train" and args.load is None:
            avg_return = last_eval_metrics["average_return"] if last_eval_metrics else 0.0
            final_checkpoint_name = (
                f"{experiment_name}__steps_final={total_num_steps}__avg={avg_return:.2f}.pt"
            )
            final_checkpoint_path = os.path.join(args.save_model_dir, final_checkpoint_name)
            torch.save(agent.policy.state_dict(), final_checkpoint_path)
            logger.log(
                "final_checkpoint",
                {
                    "checkpoint_path": final_checkpoint_path,
                    "total_steps": total_num_steps,
                    "average_return": avg_return,
                },
            )

        logger.log(
            "experiment_end",
            {
                "experiment_name": experiment_name,
                "total_steps": total_num_steps,
                "train_episodes_completed": train_num_episodes,
                "elapsed_time_sec": time.time() - start_time,
            },
        )

    if tensorboard_writer is not None:
        tensorboard_writer.close()

    env.close()


if __name__ == "__main__":
    main()