import json
import os
from models import Thought
from termcolor import colored
from openai import OpenAI
from typing import Dict, Any

class MindLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_to_file(self, filename: str, data: Dict[str, Any]):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'a') as f:
            json_str = json.dumps(data)
            f.write(json_str + '\n')
            f.flush()  # Ensure immediate writing to file


class MindComponent:
    def __init__(self, name: str, client: OpenAI):
        self.name = name
        self.client = client

    async def generate_thought(self, context: Dict) -> Thought:
        """Generate a thought based on current context"""
        print(colored(f"{self.name.title()} component generating thought...", "cyan"))
        prompt = self.create_prompt(context)
        print(colored(f"  Using prompt: {prompt}", "cyan", attrs=["dark"]))

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format=Thought
        )

        thought_content = completion.choices[0].message.parsed
        print(colored(f"  Generated thought: {thought_content.content}", "cyan"))
        return thought_content

    def get_system_prompt(self) -> str:
        """Provide the system prompt for the AI model."""
        return (
            "You are an AI assistant designed to help users generate thoughtful responses based on the provided context. "
            "Your response should be in JSON format and match the following model: "
            '{"content": "<thought_content>", "source": "<source>", "intensity": <intensity>, "emotion": "<emotion>", "associations": ["<association1>", "<association2>"]}.'
        )

    def create_prompt(self, context: Dict) -> str:
        """Create prompt based on current context"""
        # Example implementation: Convert context dictionary to a string prompt
        prompt = " ".join(f"{key}: {value}" for key, value in context.items())
        return prompt

