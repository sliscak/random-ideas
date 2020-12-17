import gym
env = gym.make('procgen:procgen-coinrun-v0')
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        break

# from procgen import ProcgenGym3Env
# env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=0, num_levels=1)