import os
from termcolor import colored
from openai import AsyncOpenAI
from datetime import datetime
import numpy as np

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

class HierarchicalMemory:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Three-tier memory system
        self.immediate_context = []  # Last few messages
        self.short_term_memory = []  # Recent summaries
        self.long_term_memory = []  # Important embeddings

        print(colored("Hierarchical memory system initialized.", "green"))
    
    async def create_embedding(self, text):
        """
        Creates an embedding for the given text.

        Args:
            text (str): The input text.

        Returns:
            numpy.ndarray: The embedding of the input text.
        """
        try:
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(colored(f"Error creating embedding: {str(e)}", "red"))
            return None

    async def assess_importance(self, message):
        """
        Assess the importance of a given message.

        Uses a GPT model to rate the importance of the given message on a scale of 0 to 1.

        Args:
            message (dict): The message to be assessed.

        Returns:
            float: The importance rating of the message.
        """
        try:
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system", "content": "Rate the importance of this message for long-term memory on a scale of 0 to 1. Respond with only the number."},
                    {"role": "user", "content": message["content"]}
                ]
            )
            return float(response.choices[0].message.content.strip())
        except Exception as e:
            print(colored(f"Error assessing importance: {str(e)}", "red"))
            return 0.0

    async def create_summary(self, messages):
        """
        Creates a concise summary of a conversation chunk.

        Args:
            messages (list[dict]): The messages to be summarized.

        Returns:
            str: The summary of the conversation chunk.
        """
        try:
            messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Create a concise summary of this conversation chunk."},
                    {"role": "user", "content": messages_text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(colored(f"Error creating summary: {str(e)}", "red"))
            return None

    async def add_memory(self, message):
        """
        Adds a new message to the memory system.

        First, the message is given a timestamp. Then, it is added to the immediate context.
        If the immediate context grows too large, the oldest messages are converted to short-term
        memory items. If the short-term memory grows too large, the oldest items are processed
        for potential long-term storage based on their importance. Finally, the long-term memory
        is trimmed to its maximum size if necessary.

        :param message: The message to be added to the memory system.
        :type message: dict
        """
        # Add timestamp
        message["timestamp"] = datetime.now().isoformat()

        # Update immediate context
        self.immediate_context.append(message)
        if len(self.immediate_context) > IMMEDIATE_CONTEXT_SIZE:
            overflow_messages = self.immediate_context[:-IMMEDIATE_CONTEXT_SIZE]
            self.immediate_context = self.immediate_context[-IMMEDIATE_CONTEXT_SIZE:]

            # Convert overflow to short-term memory
            if overflow_messages:
                summary = await self.create_summary(overflow_messages)
                if summary:
                    self.short_term_memory.append({
                        "summary": summary,
                        "messages": overflow_messages,
                        "timestamp": overflow_messages[-1]["timestamp"]
                    })

                    # Maintain short-term memory size
                    if len(self.short_term_memory) > SHORT_TERM_SIZE:
                        overflow_summaries = self.short_term_memory[:-SHORT_TERM_SIZE]
                        self.short_term_memory = self.short_term_memory[-SHORT_TERM_SIZE:]

                        # Process important entries for long-term storage
                        for summary_item in overflow_summaries:
                            importance = await self.assess_importance({"content": summary_item["summary"]})

                            if importance >= IMPORTANCE_THRESHOLD:
                                embedding = await self.create_embedding(summary_item["summary"])
                                if embedding:
                                    self.long_term_memory.append({
                                        "summary": summary_item["summary"],
                                        "embedding": embedding,
                                        "importance": importance,
                                        "timestamp": summary_item["timestamp"]
                                    })

                        # Maintain long-term memory size
                        if len(self.long_term_memory) > LONG_TERM_SIZE:
                            self.long_term_memory.sort(key=lambda x: x["importance"], reverse=True)
                            self.long_term_memory = self.long_term_memory[:LONG_TERM_SIZE]

        print(colored(f"Memory added successfully!", "green"))

    async def get_relevant_memories(self, query):
        """
        Retrieves relevant memories from the immediate context, long-term memory, and short-term memory for the given query.

        Args:
            query (str): The query to search for relevant memories.

        Returns:
            list: A list of dictionaries containing the relevant memories. Each dictionary contains the key-value pairs:
                - role (str): The role of the speaker. Set to "assistant".
                - content (str): The relevant memory content.
        """
        relevant_memories = []

        # include inmediate context
        relevant_memories.extend(self.immediate_context)

        # Find embeddings for long-term memory
        query_embedding = await self.create_embedding(query)
        if query_embedding:
            # Find relevant long-term memories
            similarities = []
            for memory in self.long_term_memory:
                similarity = np.dot(query_embedding, memory["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory["embedding"])
                )
                similarities.append((similarity, memory))

            # Sort by similarity and take top 3
            similarities.sort(key=lambda x: x[0], reverse=True)
            for sim, memory in similarities[:3]:
                if sim > 0.9:  # Similarity threshold 0.7 ?
                    relevant_memories.append({
                        "role": "assistant",
                        "content": f"Relevant past context: {memory['summary']}"
                    })

        # Get relevant short-term memories using GPT
        if self.short_term_memory:
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Select indices of relevant summaries for the query. Return space-separated numbers only."},
                    {"role": "user", "content": f"Query: {query}\n\nSummaries:\n" + "\n".join([f"{i}: {m['summary']}" for i, m in enumerate(self.short_term_memory)])}
                ]
            )

            try:
                indices = [int(i) for i in response.choices[0].message.content.split() if i.isdigit()]
                for idx in indices[:2]:  # Top 2 most relevant
                    if idx < len(self.short_term_memory):
                        relevant_memories.append({
                            "role": "assistant",
                            "content": f"Recent context: {self.short_term_memory[idx]['summary']}"
                        })
            except ValueError:
                print(colored("Error parsing relevant summary indices.", "yellow"))

        return relevant_memories

    def get_memory_stats(self):
        return {
            "immediate_context": len(self.immediate_context),
            "short_term_memory": len(self.short_term_memory),
            "long_term_memory": len(self.long_term_memory)
        }
