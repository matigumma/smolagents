import json
import os
from openai import OpenAI
from typing import Dict, Any, List
import time
from typing import List

from termcolor import colored

from components import MindLogger
from beliefs import BeliefSystem
from conclusions import ConclusionGenerator
from controllers import EmotionalProcessor, QuestionGenerator, RationalAnalyzer
from models import ConsciousState, EmotionalState, Question, Thought
from ms import SAVE_DIR, MemorySystem



class Mind:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.logger = MindLogger(SAVE_DIR)
        self.components = {
            'emotional': EmotionalProcessor('emotional', self.client),
            'rational': RationalAnalyzer('rational', self.client),
            'memory': MemorySystem('memory', self.client, self.logger),
            'curiosity': QuestionGenerator('curiosity', self.client),
            'belief': BeliefSystem('belief', self.client, self.logger),
            'conclusion': ConclusionGenerator('conclusion', self.client, self.logger),
        }
        self.conscious_state = ConsciousState(
            active_thoughts=[],
            dominant_emotion=EmotionalState.NEUTRAL,
            attention_focus="idle",
            arousal_level=0.5
        )
        self.questions: List[Question] = []
        self.initial_situation = None

    def log_thought(self, thought: Thought):
        thought_data = thought.to_dict()
        self.logger.log_to_file('thoughts.jsonl', thought_data)

    def log_question(self, question: Question):
        question_data = question.to_dict()
        self.logger.log_to_file('questions.jsonl', question_data)
    
    async def generate_new_question(self) -> str:
        """Generates a new question based in the current state"""
        context = {
            'active_thoughts': self.conscious_state.active_thoughts,
            'situation': self.conscious_state.attention_focus,
            'emotion': self.conscious_state.dominant_emotion,
            'initial_situation': self.initial_situation
        }


        question = await self.components['curiosity'].generate_question(context)

        self.questions.append(question)
        self.log_question(question)
        return question.content
    
    async def process_situation(self, situation: str):
        print(colored(f"\n> Processing situation: {situation}", "magenta", attrs=["bold"]))
        print(colored("=" * 50, "magenta"))

        context = {
            'situation': situation,
            'current_emotion': self.conscious_state.dominant_emotion,
            'arousal_level': self.conscious_state.arousal_level
        }

        print(colored(f"\n Current state", "blue"))
        print(colored(f"\n  L Emotion: {self.conscious_state.dominant_emotion}", "blue"))
        print(colored(f"\n  L Arousal: {self.conscious_state.arousal_level}", "blue"))

        print(colored(f"\n Generating component responses", "green"))
        emotional_thought = await self.components['emotional'].generate_thought(context)
        time.sleep(1)
        rational_thought = await self.components['rational'].generate_thought(context)

        self.log_thought(emotional_thought)
        self.log_thought(rational_thought)

        print(colored(f"\n Storing new thoughts on memory", "yellow"))
        memory_system = self.components['memory']
        await memory_system.store_memory(emotional_thought)
        await memory_system.store_memory(rational_thought)

        relevant_memories = await memory_system.retrieve_relevant_memories(context, num_memories=3, similarity_threshold=0.7)

        print(colored(f"\n Updating conscious state", "magenta"))

        self.conscious_state.active_thoughts = [emotional_thought, rational_thought] + relevant_memories
        old_emotion = self.conscious_state.dominant_emotion
        self.conscious_state.dominant_emotion = self.determine_dominant_emotion()
        self.conscious_state.attention_focus = situation

        print(colored(f"\n Updated state", "blue"))
        print(colored(f"\n  L Emotion: {old_emotion} -> {self.conscious_state.dominant_emotion}", "blue"))
        print(colored(f"\n  L Attention: {self.conscious_state.attention_focus}", "blue"))
        print(colored(f"\n  L Active Thoughts: {self.conscious_state.active_thoughts}", "blue"))
        print(colored("=" * 50 + "\n", "magenta"))

        belief_context = {
            'active_thoughts': self.conscious_state.active_thoughts,
            'situation': situation,
            'emotion': self.conscious_state.dominant_emotion,
        }

        await self.components['belief'].evaluate_beliefs(belief_context)

    def determine_dominant_emotion(self):
        """Determine dominant emotion based on active thoughts"""
        if not self.conscious_state.active_thoughts:
            return EmotionalState.NEUTRAL
        
        emotions_count ={}
        max_intensity = 0
        dominant_emotion = EmotionalState.NEUTRAL

        for thought in self.conscious_state.active_thoughts:
            if thought.intensity > max_intensity:
                max_intensity = thought.intensity
                dominant_emotion = thought.emotion

            if thought.emotion in emotions_count:
                emotions_count[thought.emotion] += 1
            else:
                emotions_count[thought.emotion] = 1
        
        return dominant_emotion

    async def explore(self, situation: str):
        await self.process_situation(situation)
        await self.generate_new_question()

        while True:
            user_input = input("Enter a response (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            await self.process_situation(user_input)
            await self.generate_new_question()

        print("Goodbye!")

    async def run(self):
        await self.explore(self.initial_situation)