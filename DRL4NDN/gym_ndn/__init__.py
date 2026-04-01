from gymnasium.envs.registration import register

register(
    id="NDNRouter-v0",
    entry_point="gym_ndn.envs:NDNRouterEnv",
    max_episode_steps=1000,
)
