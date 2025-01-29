import time
from typing import Dict
from litellm import OpenAI
from termcolor import colored
from components import MindComponent, MindLogger
from models import Belief


class BeliefSystem(MindComponent):
    def __init__(self, name: str, client: OpenAI, logger: 'MindLogger'):
        super().__init__(name, client)
        self.beliefs: list[Belief] = []
        self.logger = logger

    def _get_system_prompt(self) -> str:
        return """You are the belief formation center of a mind. Analyze thoughts and form
        coherent beliefs and conclusions. Consider evidence both for and against each belief."""

    def _create_prompt(self, context: Dict) -> str:
        thoughts = context.get('active_thoughts', [])
        thought_contents = [t.content for t in thoughts] if thoughts else []
        existing_beliefs = [b.statement for b in self.beliefs] if self.beliefs else []

        return f"""Based on these thoughts: {thought_contents}
        And existing beliefs: {existing_beliefs}
        Analyze the evidence and form or update a belief/conclusion."""

    async def evaluate_beliefs(self, context: Dict):
        """Evaluate current thoughts and update beliefs"""
        print(colored("\n🐾 evaluating beliefs...", "blue"))
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Belief
        )

        new_belief = completion.choices[0].message.parsed
        self._update_beliefs(new_belief)
        self.logger.log_to_file('beliefs.jsonl', new_belief.to_dict())

    def _update_beliefs(self, new_belief: Belief):
        """Update existing beliefs or add new ones"""

        # Find similar existing beliefs
        similar_beliefs = [b for b in self.beliefs
                        if self._belief_similarity(b.statement, new_belief.statement) > 0.8]

        if similar_beliefs:
            # Update existing belief
            existing_belief = similar_beliefs[0]
            # Adjust confidence based on new evidence
            existing_belief.confidence = (existing_belief.confidence + new_belief.confidence) / 2
            existing_belief.supporting_thoughts.extend(new_belief.supporting_thoughts)
            existing_belief.counter_thoughts.extend(new_belief.counter_thoughts)
            existing_belief.last_updated = time.time()
            # Increase stability with each confirmation
            existing_belief.stability = min(1.0, existing_belief.stability + 0.1)

        else:
            # Add new belief
            new_belief.last_updated = time.time()
            new_belief.stability = 0.1  # Start with low stability
            self.beliefs.append(new_belief)

    def _belief_similarity(self, belief1: str, belief2: str) -> float:
        """Simple similarity check - can be enhanced with embeddings"""

        # This is a placeholder - you might want to use embedding similarity here
        words1 = set(belief1.lower().split())
        words2 = set(belief2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0

    def _log_belief(self, belief: Belief):
        """Log the belief"""
        belief_data = belief.to_dict()
        self.logger.log_beliefs('beliefs.jsonl', belief_data)