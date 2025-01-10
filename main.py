import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from huggingface_hub import login

hf_api_key = os.getenv("HF_API_KEY")
if hf_api_key is None:
    raise EnvironmentError("HF_API_KEY environment variable not set.")
login(hf_api_key)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
