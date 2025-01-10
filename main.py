import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from huggingface_hub import login

login(os.getenv("HF_API_KEY"))

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
