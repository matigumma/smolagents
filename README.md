# Playing with smolagents

Este repositorio contiene un script de Python que implementa un sistema multiagente. Utiliza varios modelos y herramientas 
El script aprovecha los modelos de Hugging Face y el modelo LiteLLM de OpenAI para crear agentes capaces de ejecutar codigo creado por el mismo agente.

## Features

- **Hugging Face Integration**: Utilizes Hugging Face's `CodeAgent` and `HfApiModel` for executing code-related tasks.
- **OpenAI Integration**: Implements a proxy for OpenAI's LiteLLM model to handle requests.
- **Custom Tool Creation**: Demonstrates how to create and use custom tools with the `@tool` decorator.
- **Web Scraping**: Includes an example of an agent that can scrape and return data from a webpage in JSON format.
- **CodeAgent**: Test
- **ToolCallingAgent**: Test
- **Search Functionality**: Integrates a search tool to fetch information from DuckDuckGo.


## How?
1. UV package manager:
   [UV install](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

1. Prepare environment:
   ```bash
   uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API tokens:
   ```plaintext
   HF_API_TOKEN=your_hugging_face_api_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

Run the script using:

```bash
python main.py
```


