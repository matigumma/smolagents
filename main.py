from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from huggingface_hub import login
from dotenv import load_dotenv  # Add this import at the top
import os

load_dotenv()

hf_api_key = os.getenv("HF_API_TOKEN")
if hf_api_key is None:
    raise EnvironmentError("HF_API_TOKEN environment variable not set.")
login(hf_api_key)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
