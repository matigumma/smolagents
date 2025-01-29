```mermaid
mindmap
  root((Memory System))
    Memory Components
      MemorySystem
        Load Existing Memories
          MEMORY_LOG
          EMBEDDING_LOG
        Store Memories
          Thought Embedding
          Semantic Similarity
        Retrieve Memories
          Context Embedding
          Similarity Calculation
    Memory Flow
      Mind
        Components
          Emotional Processor
          Rational Analyzer
          Memory System
          Belief System
          Conclusion Generator
        Process Situation
          Generate Thoughts
          Log Thoughts
          Store in Memory
    Key Concepts
      Thought
        Content
        Source
        Intensity
        Emotion
        Associations
      Conscious State
        Active Thoughts
        Dominant Emotion
        Attention Focus
        Arousal Level
    Persistence Mechanisms
      MindLogger
        Log to File
      Embedding Cache
        Store Embeddings
```
## Memory System Workflow

```mermaid
flowchart TD
    A[Mind Processes Situation] --> B{Generate Thoughts}
    B -->|Emotional Thought| C[Create Thought Object]
    B -->|Rational Thought| C
    
    C --> D[Memory System]
    D --> E[Generate Embedding]
    E --> F{Check Embedding Cache}
    
    F -->|Not Cached| G[Call OpenAI Embedding API]
    G --> H[Store in Embedding Cache]
    F -->|Cached| I[Retrieve Cached Embedding]
    
    C --> J[Store Thought in Memory List]
    J --> K[Log to MEMORY_LOG]
    
    E --> L[Semantic Similarity Calculation]
    L --> M{Retrieve Relevant Memories}
    M --> N[Filter by Similarity Threshold]
    N --> O[Rank and Select Top Memories]
    
    O --> P[Return Relevant Memories]
    P --> Q[Use in Further Processing]
    
    subgraph Memory Persistence
        K
        H
    end
    
    subgraph Memory Retrieval
        L
        M
        N
        O
    end
```

## Memory Interaction Sequence

```mermaid
sequenceDiagram
    participant Mind
    participant MemorySystem
    participant OpenAI
    participant MindLogger
    
    Mind->>MemorySystem: Store Thought
    MemorySystem->>OpenAI: Generate Embedding
    OpenAI-->>MemorySystem: Return Embedding
    MemorySystem->>MemorySystem: Cache Embedding
    MemorySystem->>MindLogger: Log Memory
    
    Mind->>MemorySystem: Retrieve Relevant Memories
    MemorySystem->>OpenAI: Embed Context
    OpenAI-->>MemorySystem: Return Context Embedding
    MemorySystem->>MemorySystem: Calculate Similarities
    MemorySystem-->>Mind: Return Top Relevant Memories
```

### Key Workflow Explanations

1. **Thought Generation**: 
   - Mind generates emotional and rational thoughts
   - Each thought is converted to a Thought object

2. **Embedding Process**:
   - Generate embeddings using OpenAI's API
   - Cache embeddings for performance
   - Store in MEMORY_LOG

3. **Memory Retrieval**:
   - Embed current context
   - Calculate semantic similarity
   - Filter and rank memories
   - Return most relevant memories

4. **Persistence Mechanisms**:
   - Log thoughts to MEMORY_LOG
   - Cache embeddings in memory
   - Support for loading existing memories on startup
