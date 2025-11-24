from rainbowrl import Agent

agent = Agent(environment="ALE/Pong-v5", training_starts=2000)
data = agent.loop(verbose=True)

print(data.loss, data.hns, data.rewards)
