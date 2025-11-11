# import gymnasium as gym

# env = gym.make("Ant-v2")
# env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# # The following always has to hold:
# assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
# assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
# assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# # However goals can also be substituted:
# substitute_goal = obs["achieved_goal"].copy()
# substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
# substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
# substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)

import json

data = json.load(open("/home/qhp/project/deep_rl/logs/env=HalfCheetah-v5__algo=trpo__phase=train__seed=26__backtrack_alpha=0.5__backtrack_coeff=0.8__backtrack_iter=10__delta=0.01__gamma=0.99__lam=0.97__sample_size=2048__train_vf_iters=120__vf_lr=0.001.json"))
print(data["events"])