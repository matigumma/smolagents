from typing import Dict
from mind import MindComponent
from models import Question
from termcolor import colored

class EmotionalProcessor(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the emotional processing center of a mind.
        Generate emotional responses and associated thoughts based on current context."""

    def _create_prompt(self, context: Dict) -> str:
        return f"Given the current situation: {context.get('situation', '')}, " \
               f"and current emotional state: {context.get('current_emotion', '')}, " \
               f"generate an emotional thought response."

class RationalAnalyzer(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the rational analysis center of a mind.
        Generate logical thoughts and analytical observations based on current context."""

    def _create_prompt(self, context: Dict) -> str:
        return f"Analyze this situation logically: {context.get('situation', '')}"

class QuestionGenerator(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the curiosity center of a mind. Generate meaningful questions
        based on current thoughts, context, and the initial exploration topic. Focus on deep,
        exploratory questions that build upon previous insights."""

    def _create_prompt(self, context: Dict) -> str:
        thoughts = context.get('active_thoughts', [])
        thought_contents = [t.content for t in thoughts] if thoughts else []
        initial_situation = context.get('initial_situation', '')

        return f"""Initial exploration topic: {initial_situation}
        Based on the current thoughts: {thought_contents}
        and situation: {context.get('situation', '')},
        generate a question that helps explore and build upon our understanding of the initial topic."""
    
    async def generate_question(self, context:Dict) -> Question:
        print(colored(f"\n ❓ generating question...", "magenta"))

        completion = self.client.beta.chat.completions.create(
            model="gpt-4-mini",
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Question
        )

        question = completion.choices[0].message.parsed
        # question = Question(question_content)
        print(colored(f"  ⌙ Generated question: {question.content}", "magenta"))
        return question