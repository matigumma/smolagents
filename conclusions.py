from typing import Dict
from litellm import OpenAI
from termcolor import colored
from components import MindComponent, MindLogger
from models import Conclusion


class ConclusionGenerator(MindComponent):
    def __init__(self, name: str, client: OpenAI, logger: 'MindLogger'):
        super().__init__(name, client)
        self.logger = logger

    def _get_system_prompt(self) -> str:
        return """You are the conclusion generation center of a mind. Synthesize current beliefs, thoughts, and emotional state into meaningful conclusions about the mind's current understanding."""

    def _create_prompt(self, context: Dict) -> str:
        beliefs = context.get('beliefs', [])
        thoughts = context.get('active_thoughts', [])
        belief_statements = [b.statement for b in beliefs] if beliefs else []
        thought_contents = [t.content for t in thoughts] if thoughts else []

        return f"""Based on current beliefs: {belief_statements}
And active thoughts: {thought_contents}
And emotional state: {context.get('emotion', 'NEUTRAL')}
Generate a conclusive statement about the current understanding."""

    async def generate_conclusion(self, context: Dict) -> Conclusion:
        """Generate a conclusion based on current mental state"""
        print(colored("\n> Generating conclusion...", "green"))

        completion = self.client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Conclusion
        )