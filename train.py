from rainbowrl import Agent

agent = Agent("Pong-v5")
data = agent.loop(verbose=True)

print(data.loss, data.hns, data.rewards)
