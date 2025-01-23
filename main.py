from dotenv import load_dotenv  # Add this import at the top
import os

# huggingface code interpreter based agent system
from smolagents import CodeAgent, ManagedAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, HfApiModel # default model = Qwen/Qwen2.5-Coder-32B-Instruct # for free
from huggingface_hub import login

# local imports
# from tools import save_image_to_file, image_generation, print_chinese



load_dotenv()

hf_api_key = os.getenv("HF_API_TOKEN")
if hf_api_key is None:
    raise EnvironmentError("HF_API_TOKEN environment variable not set.")

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

login(hf_api_key)

## default huggingface agent - qwen to response - with default tools (DuckduckGoSearchTool, Python Code Interpreter, STT Audio Transcriber)

# default_qwen_agent = CodeAgent(tools=[], model=HfApiModel(), add_base_tools=True)

# default_qwen_agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

## openai over LiteLLM proxy - gpt-4o-mini to response

openai_model = LiteLLMModel(model_id="gpt-4o-mini")

# openai_agent = CodeAgent(tools=[], model=openai_model, add_base_tools=True)

# openai_agent.run("Hay 2 patos y 3 patas en un corral. Cuantas patas hay en total?")

## example agent with authorized tool packages imports

# authorized_packages_agent = CodeAgent(
#     tools=[], model=HfApiModel(), additional_authorized_imports=["requests", "bs4", "json"]
# )

# authorized_packages_agent.run("Give me content output as JSON of a webpage find_all li and for each li give me the values of inside elements with class '.prop-desc-tipo-ub', '.prop-desc-dir', '.prop-valor-nro' (example xpath of first element: //html/body/li[1]/div/text() !important split by \n and get first slice only), '.codref', '.prop-data':nth-child(1), 'prop-data2':nth-child(1) for a given url: https://www.cristiangonzalezpropiedades.com.ar/Propiedades?q=&currency=ANY&min-price=&max-price=&min-roofed=&max-roofed=&min-surface=&max-surface=&min-total_surface=&max-total_surface=&min-front_measure=&max-front_measure=&min-depth_measure=&max-depth_measure=&age=&min-age=&max-age=&suites=&rooms=&tags=&operation=&locations=&location_type=&ptypes=&o=&watermark=&p=2")

## example useless tool - understand how to parse arguments, process, return values and utilize with agents

# agent_with_system_prompt = ToolCallingAgent(tools=[print_chinese.run], model=HfApiModel())

# agent_with_system_prompt.run("Printout this translated sentence into Chinese: Hello world!")

## test tool alone # it print search results...

# search_tool = DuckDuckGoSearchTool()
# print(search_tool("Who's the current president of United States?"))

# agent_with_imported_tool = CodeAgent(tools=[image_generation.run, save_image_to_file.run], model=HfApiModel())

# agent_with_imported_tool.run("Generate an image of a Red Fox and then save the content of the image generated into a file")

agent_with_aditional_params = CodeAgent(
    tools=[], model=openai_model, agent_with_aditional_params={"image": './imagen.jpg'}
)

agent_with_aditional_params.run("send an image to the model and ask which color of eyes have the fox in the image?")
