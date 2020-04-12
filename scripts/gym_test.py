import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()