import os
import json
from termcolor import colored
# from openai import AsyncOpenAI
# from datetime import datetime
# import numpy as np
from typing import Dict, Any, List

from models import Thought

# Hierarchical Memory System
# This is the most sophisticated memory system, implementing a three-tier approach
# that combines immediate context, short-term summaries, and long-term embeddings.

# How it works:
# 1. Three-tiered memory structure:
#    - Immediate context: Last few messages [IMMEDIATE_CONTEXT_SIZE]
#    - Short-term memory: Summaries of recent conversations [SHORT_TERM_SIZE]
#    - Long-term memory: Important embedded memories [LONG_TERM_SIZE]

# 2. Memory flow:
#    a. New messages go to immediate context.
#    b. Overflow from immediate context is summarized into short-term memory.
#    c. Important summaries are embedded and stored in long-term memory.

# 3. Retrieval process:
#    - Always includes immediate context.
#    - Uses embeddings to find relevant long-term memories.
#    - Uses GPT to find relevant short-term summaries.
#    - Combines all relevant information with proper context markers.

# Key features:
# - Comprehensive memory management.
# - Adaptive memory flow between tiers.
# - Importance-based filtration.
# - Automatic memory retrieval.

# Use cases:
# - Applications requiring both detailed recent context and historical information.
# - Long-running conversations.
# - Situations where memory organization is critical.

# Constants
MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
IMMEDIATE_CONTEXT_SIZE = 5
SHORT_TERM_SIZE = 20
LONG_TERM_SIZE = 100
IMPORTANCE_THRESHOLD = 0.7
MAX_ITERATIONS = None
SLEEP_DURATION = 2 # in seconds

SAVE_DIR="mind_logs"
THOUGHT_LOG=f"{SAVE_DIR}/thoughts.jsonl"
MEMORY_LOG=f"{SAVE_DIR}/memory.jsonl"
EMBEDDING_LOG=f"{SAVE_DIR}/embeddings.jsonl"
STATE_LOG=f"{SAVE_DIR}/state.jsonl"
QUESTION_LOG=f"{SAVE_DIR}/questions.jsonl"
BELIEF_LOG=f"{SAVE_DIR}/beliefs.jsonl"
CONCLUSION_LOG=f"{SAVE_DIR}/conclusions.jsonl"
CONCLUSION_INTERVAL = 5 


class MemorySystem:
    def __init__(self, name: str, client, logger):
        self.name = name
        self.client = client
        self.logger = logger
        self.memories: List['Thought'] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._load_existing_memories()

    def _load_existing_memories(self):
        """Load existing memories and their embeddings from files"""
        try:
            # Load memories
            if os.path.exists("MEMORY_LOG"):
                with open("MEMORY_LOG", 'r') as f:
                    for line in f:
                        memory_data = json.loads(line)
                        thought = Thought(memory_data)
                        self.memories.append(thought)

            # Load embeddings
            if os.path.exists("EMBEDDING_LOG"):
                with open("EMBEDDING_LOG", 'r') as f:
                    for line in f:
                        embedding_data = json.loads(line)
                        self.embeddings_cache[embedding_data['content']] = embedding_data['embedding']

            print(colored(f"Loaded {len(self.memories)} memories and {len(self.embeddings_cache)} embeddings", "green"))
        except Exception as e:
            print(colored(f"Error loading memories: {e}", "red"))

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    async def store_memory(self, thought: 'Thought'):
        """Store memory and its embedding"""
        print(colored(f"Storing memory: {thought.content[:50]}...", "yellow"))

        # Get embedding for the thought content
        embedding = await self.get_embedding(thought.content)
        self.embeddings_cache[thought.content] = embedding

        # Store the memory
        self.memories.append(thought)
        memory_data = thought.to_dict()
        self.logger.log_to_file("memories.json", memory_data)

        # Store the embedding
        embedding_data = {
            'content': thought.content,
            'embedding': embedding
        }
        self.logger.log_to_file("embeddings.json", embedding_data)

    async def retrieve_relevant_memories(self, context: Dict, num_memories: int = 3, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity"""
        print(colored("\n 🔎 Searching for relevant memories...", "yellow")) 
        if not self.memories:
            return []
        
        # Create a combined context string
        context_string = f"{context.get('situation', '')} {context.get('current_emotion', '')}"
        if 'active_thoughts' in context:
            thought_contents = [t.content for t in context['active_thoughts']]
            context_string += ' ' + ' '.join(thought_contents)

        print(colored(f" L Context: {context_string[:100]}...", "yellow", attrs=["dark"]))

        # Get embedding for the context
        context_embedding = await self.get_embedding(context_string)

        # Calculate similarities and sort memories
        similarities = []
        for memory in self.memories:
            # Get or calculate memory embedding
            if memory.content not in self.embeddings_cache:
                memory_embedding = await self.get_embedding(memory.content)
                self.embeddings_cache[memory.content] = memory_embedding
            else:
                memory_embedding = self.embeddings_cache[memory.content]

            # Calculate similarity
            similarity = self._cosine_similarity(context_embedding, memory_embedding)
            similarities.append((memory, similarity))

        # Filter memories based on similarity threshold
        filtered_memories = [mem for mem, sim in similarities if sim >= similarity_threshold]
        
        # Sort by similarity and return top N memories
        sorted_memories = sorted(filtered_memories, key=lambda x: x[1], reverse=True)[:num_memories]
        return [mem[0] for mem in sorted_memories]

# class HierarchicalMemory:
#     def __init__(self):
#         self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         # Three-tier memory system
#         self.immediate_context = []  # Last few messages
#         self.short_term_memory = []  # Recent summaries
#         self.long_term_memory = []  # Important embeddings

#         print(colored("Hierarchical memory system initialized.", "green"))
    
#     async def create_embedding(self, text):
#         """
#         Creates an embedding for the given text.

#         Args:
#             text (str): The input text.

#         Returns:
#             numpy.ndarray: The embedding of the input text.
#         """
#         try:
#             response = await self.client.embeddings.create(
#                 model=EMBEDDING_MODEL,
#                 input=text
#             )
#             return response.data[0].embedding
#         except Exception as e:
#             print(colored(f"Error creating embedding: {str(e)}", "red"))
#             return None

#     async def assess_importance(self, message):
#         """
#         Assess the importance of a given message.

#         Uses a GPT model to rate the importance of the given message on a scale of 0 to 1.

#         Args:
#             message (dict): The message to be assessed.

#         Returns:
#             float: The importance rating of the message.
#         """
#         try:
#             response = await self.client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {
#                         "role": "system", "content": "Rate the importance of this message for long-term memory on a scale of 0 to 1. Respond with only the number."},
#                     {"role": "user", "content": message["content"]}
#                 ]
#             )
#             return float(response.choices[0].message.content.strip())
#         except Exception as e:
#             print(colored(f"Error assessing importance: {str(e)}", "red"))
#             return 0.0

#     async def create_summary(self, messages):
#         """
#         Creates a concise summary of a conversation chunk.

#         Args:
#             messages (list[dict]): The messages to be summarized.

#         Returns:
#             str: The summary of the conversation chunk.
#         """
#         try:
#             messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
#             response = await self.client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": "Create a concise summary of this conversation chunk."},
#                     {"role": "user", "content": messages_text}
#                 ]
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print(colored(f"Error creating summary: {str(e)}", "red"))
#             return None

#     async def add_memory(self, message):
#         """
#         Adds a new message to the memory system.

#         First, the message is given a timestamp. Then, it is added to the immediate context.
#         If the immediate context grows too large, the oldest messages are converted to short-term
#         memory items. If the short-term memory grows too large, the oldest items are processed
#         for potential long-term storage based on their importance. Finally, the long-term memory
#         is trimmed to its maximum size if necessary.

#         :param message: The message to be added to the memory system.
#         :type message: dict
#         """
#         # Add timestamp
#         message["timestamp"] = datetime.now().isoformat()

#         # Update immediate context
#         self.immediate_context.append(message)
#         if len(self.immediate_context) > IMMEDIATE_CONTEXT_SIZE:
#             overflow_messages = self.immediate_context[:-IMMEDIATE_CONTEXT_SIZE]
#             self.immediate_context = self.immediate_context[-IMMEDIATE_CONTEXT_SIZE:]

#             # Convert overflow to short-term memory
#             if overflow_messages:
#                 summary = await self.create_summary(overflow_messages)
#                 if summary:
#                     self.short_term_memory.append({
#                         "summary": summary,
#                         "messages": overflow_messages,
#                         "timestamp": overflow_messages[-1]["timestamp"]
#                     })

#                     # Maintain short-term memory size
#                     if len(self.short_term_memory) > SHORT_TERM_SIZE:
#                         overflow_summaries = self.short_term_memory[:-SHORT_TERM_SIZE]
#                         self.short_term_memory = self.short_term_memory[-SHORT_TERM_SIZE:]

#                         # Process important entries for long-term storage
#                         for summary_item in overflow_summaries:
#                             importance = await self.assess_importance({"content": summary_item["summary"]})

#                             if importance >= IMPORTANCE_THRESHOLD:
#                                 embedding = await self.create_embedding(summary_item["summary"])
#                                 if embedding:
#                                     self.long_term_memory.append({
#                                         "summary": summary_item["summary"],
#                                         "embedding": embedding,
#                                         "importance": importance,
#                                         "timestamp": summary_item["timestamp"]
#                                     })

#                         # Maintain long-term memory size
#                         if len(self.long_term_memory) > LONG_TERM_SIZE:
#                             self.long_term_memory.sort(key=lambda x: x["importance"], reverse=True)
#                             self.long_term_memory = self.long_term_memory[:LONG_TERM_SIZE]

#         print(colored(f"Memory added successfully!", "green"))

#     async def get_relevant_memories(self, query):
#         """
#         Retrieves relevant memories from the immediate context, long-term memory, and short-term memory for the given query.

#         Args:
#             query (str): The query to search for relevant memories.

#         Returns:
#             list: A list of dictionaries containing the relevant memories. Each dictionary contains the key-value pairs:
#                 - role (str): The role of the speaker. Set to "assistant".
#                 - content (str): The relevant memory content.
#         """
#         relevant_memories = []

#         # include inmediate context
#         relevant_memories.extend(self.immediate_context)

#         # Find embeddings for long-term memory
#         query_embedding = await self.create_embedding(query)
#         if query_embedding:
#             # Find relevant long-term memories
#             similarities = []
#             for memory in self.long_term_memory:
#                 similarity = np.dot(query_embedding, memory["embedding"]) / (
#                     np.linalg.norm(query_embedding) * np.linalg.norm(memory["embedding"])
#                 )
#                 similarities.append((similarity, memory))

#             # Sort by similarity and take top 3
#             similarities.sort(key=lambda x: x[0], reverse=True)
#             for sim, memory in similarities[:3]:
#                 if sim > 0.9:  # Similarity threshold 0.7 ?
#                     relevant_memories.append({
#                         "role": "assistant",
#                         "content": f"Relevant past context: {memory['summary']}"
#                     })

#         # Get relevant short-term memories using GPT
#         if self.short_term_memory:
#             response = await self.client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": "Select indices of relevant summaries for the query. Return space-separated numbers only."},
#                     {"role": "user", "content": f"Query: {query}\n\nSummaries:\n" + "\n".join([f"{i}: {m['summary']}" for i, m in enumerate(self.short_term_memory)])}
#                 ]
#             )

#             try:
#                 indices = [int(i) for i in response.choices[0].message.content.split() if i.isdigit()]
#                 for idx in indices[:2]:  # Top 2 most relevant
#                     if idx < len(self.short_term_memory):
#                         relevant_memories.append({
#                             "role": "assistant",
#                             "content": f"Recent context: {self.short_term_memory[idx]['summary']}"
#                         })
#             except ValueError:
#                 print(colored("Error parsing relevant summary indices.", "yellow"))

#         return relevant_memories

#     def get_memory_stats(self):
#         return {
#             "immediate_context": len(self.immediate_context),
#             "short_term_memory": len(self.short_term_memory),
#             "long_term_memory": len(self.long_term_memory)
#         }
