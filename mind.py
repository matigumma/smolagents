import json
import os
from termcolor import colored
from openai import OpenAI
from typing import Dict, Any

from models import Thought

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

        completion = self.client.beta.chat.completions.create(
            model="gpt-4-mini",
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        thought_content = completion.choices[0].message.content
        thought = Thought(thought_content)
        print(colored(f"  Generated thought: {thought.content}", "cyan"))
        return thought

    def get_system_prompt(self) -> str:
        raise NotImplementedError

    def create_prompt(self, context: Dict) -> str:
        raise NotImplementedError




# class Mind:
#     def __init__(self, openai_client: OpenAI):
#         self.client = openai_client
#         self.logger = MindLogger(SAVE_DIR)
#         self.components = {
#             'emotional': EmotionalProcessor('emotional', self.client),
#             # 'rational': RationalAnalyzer('rational', self.client),
#             'memory': MemorySystem('memory', self.client, self.logger),
#             # 'curiosity': QuestionGenerator('curiosity', self.client),
#             # 'belief': BeliefSystem('belief', self.client, self.logger),
#             # 'conclusion': ConclusionGenerator('conclusion', self.client, self.logger),
#         }
#         self.conscious_state = ConsciousState(
#             active_thoughts=[],
#             dominant_emotion=EmotionalState.NEUTRAL,
#             attention_focus="idle",
#             arousal_level=0.5
#         )
#         self.questions: List[Question] = []
#         self.initial_situation = None

#     def log_state(self):
#         state_data = {
#             'dominant_emotion': self.conscious_state.dominant_emotion,
#             'attention_focus': self.conscious_state.attention_focus,
#             'arousal_level': self.conscious_state.arousal_level,
#             'active_thoughts_count': len(self.conscious_state.active_thoughts)
#         }
#         self.logger.log_to_file('states.json', state_data)
