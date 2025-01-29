# Core data models

from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class EmotionalState(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"

class Thought(BaseModel):
    content: str = Field(description="The actual content of the thought")
    source: str = Field(description="Which part of mind generated it")
    intensity: float = Field(description="How strongly this thought presents, between 0 and 1")
    emotion: EmotionalState = Field(description="The emotional state associated with this thought")
    associations: List[str] = Field(description="Associations to other thoughts")

    def to_dict(self):
        return {
            'content': self.content,
            'source': self.source,
            'intensity': self.intensity,
            'emotion': self.emotion,
            'associations': self.associations,
        }

class Question(BaseModel):
    content: str = Field(description="The content of the question")
    source: str = Field(description="Which part of mind generated it")
    importance: float = Field(description="How important is this question, between 0 and 1")
    context: str = Field(description="The context in which this question was generated")

    def to_dict(self):
        return {
            'content': self.content,
            'source': self.source,
            'importance': self.importance,
            'context': self.context,
        }

class ConsciousState(BaseModel):
    active_thoughts: List[Thought] = Field(description="Currently active thoughts in consciousness")
    dominant_emotion: EmotionalState = Field(description="The current dominant emotion")
    attention_focus: str = Field(description="What the mind is currently focused on")
    arousal_level: float = Field(description="How aroused the mind is, between 0 and 1")

class Belief(BaseModel):
    statement: str = Field(description="The belief statement")
    confidence: float = Field(description="Confidence level in this belief (0-1)")
    supporting_thoughts: List[str] = Field(description="References to thoughts that support this belief")
    counter_thoughts: List[str] = Field(description="References to thoughts that challenge this belief")
    last_updated: float = Field(description="Timestamp of last update")
    stability: float = Field(description="Stability of belief (0-1)")
    
    def to_dict(self):
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_thoughts": self.supporting_thoughts,
            "counter_thoughts": self.counter_thoughts,
            "last_updated": self.last_updated,
            "stability": self.stability
        }

class Conclusion(BaseModel):
    statement: str = Field(description="The conclusion statement")
    confidence: float = Field(description="Confidence level in this conclusion (0-1)")
    supporting_beliefs: List[str] = Field(description="Key beliefs supporting this conclusion")
    context: str = Field(description="The context in which this conclusion was generated")
    timestamp: float = Field(description="When the conclusion was generated")
