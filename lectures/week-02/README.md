# Week 2: Building Your First AI Agent

## 📚 Lecture Overview

This week we dive deep into building AI agents using the LangChain framework. You'll learn about agent architectures, prompt engineering, and how to create sophisticated conversational AI systems.

## 🎯 Learning Objectives

By the end of this week, you will:
- Understand agent architectures and frameworks
- Master LangChain components and chains
- Learn advanced prompt engineering techniques
- Build conversational agents with memory and tools
- Implement specialized agents for specific domains

## 📖 Lecture Content

### 1. Agent Architectures

**What is an AI Agent?**
- Autonomous systems that can perceive, reason, and act
- Components: Perception, Reasoning, Memory, Action
- Types: Reactive, Deliberative, Hybrid

**Agent Frameworks:**
- LangChain: Most popular Python framework
- AutoGPT: Autonomous task execution
- BabyAGI: Task management and planning
- CrewAI: Multi-agent collaboration

**Architecture Patterns:**
```
Input → Preprocessing → LLM → Post-processing → Output
         ↓
    Memory/Context
         ↓
    Tools/External APIs
```

### 2. LangChain Deep Dive

**Core Components:**

**LLMs and Chat Models:**
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Text completion model
llm = OpenAI(temperature=0.7)

# Chat model
chat = ChatOpenAI(temperature=0.7)
```

**Prompts and Templates:**
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Text prompt
template = """
You are a helpful AI assistant. Answer the following question:

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Chat prompt
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Answer this question: {question}")
])
```

**Chains:**
```python
from langchain.chains import LLMChain, ConversationChain

# Simple chain
chain = LLMChain(llm=llm, prompt=prompt)

# Conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
```

### 3. Advanced Prompt Engineering

**Prompt Design Principles:**
- **Clarity**: Be specific and unambiguous
- **Context**: Provide relevant background information
- **Examples**: Include few-shot examples when possible
- **Constraints**: Set clear boundaries and expectations

**Prompt Patterns:**

**Chain-of-Thought:**
```
Question: What is 15 * 27?

Let me think through this step by step:
1. First, I'll multiply 15 by 20: 15 * 20 = 300
2. Then, I'll multiply 15 by 7: 15 * 7 = 105
3. Finally, I'll add them together: 300 + 105 = 405

Answer: 405
```

**Role-Based Prompts:**
```
You are an expert software architect with 20 years of experience.
Your task is to review this code and provide architectural feedback.
Focus on scalability, maintainability, and best practices.

Code to review:
{code}
```

**Few-Shot Learning:**
```
Input: "I love this movie"
Output: "positive"

Input: "This is terrible"
Output: "negative"

Input: "It was okay"
Output: "neutral"

Input: "{user_input}"
Output:
```

### 4. Memory Systems

**Types of Memory:**

**Conversation Buffer Memory:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
```

**Conversation Buffer Window Memory:**
```python
from langchain.memory import ConversationBufferWindowMemory

# Keep last 5 exchanges
memory = ConversationBufferWindowMemory(k=5)
```

**Conversation Summary Memory:**
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
```

**Vector Store Memory:**
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

### 5. Tools and External APIs

**LangChain Tools:**
```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Custom tool
def search_web(query: str) -> str:
    """Search the web for current information"""
    # Implementation here
    return f"Search results for: {query}"

# Create tool
search_tool = Tool(
    name="web_search",
    func=search_web,
    description="Search the web for current information"
)

# Initialize agent with tools
agent = initialize_agent(
    [search_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

**Built-in Tools:**
- `DuckDuckGoSearchRun`: Web search
- `WikipediaQueryRun`: Wikipedia lookups
- `PythonREPLTool`: Execute Python code
- `SerpAPIWrapper`: Google search results

### 6. Building Specialized Agents

**Code Review Agent:**
```python
class CodeReviewAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code reviewer. Analyze the code for:
            1. Code quality and best practices
            2. Potential bugs and issues
            3. Security vulnerabilities
            4. Performance considerations
            5. Suggestions for improvement"""),
            ("human", "Review this {language} code:\n\n{code}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def review(self, code: str, language: str = "Python") -> str:
        return self.chain.run(code=code, language=language)
```

**Data Analysis Agent:**
```python
class DataAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)
        self.tools = [
            Tool(
                name="pandas_analysis",
                func=self.analyze_data,
                description="Analyze data using pandas"
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def analyze_data(self, query: str) -> str:
        # Implement data analysis logic
        pass
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Review LangChain documentation
2. Set up API keys for OpenAI and other services
3. Install additional dependencies: `pip install wikipedia-api duckduckgo-search`
4. Practice with basic LangChain components

**Workshop Goals:**
- Build a conversational agent with memory
- Implement tools and external API integration
- Create a specialized agent for your domain
- Deploy the agent as a web service

## 📚 Additional Resources

**Reading Materials:**
- [LangChain Documentation](https://python.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Building LLM Applications](https://www.langchain.com/use-cases)
- [Agent Architectures](https://arxiv.org/abs/2303.07839)

**Videos:**
- [LangChain Tutorial](https://www.youtube.com/watch?v=aywZrzNaKjs)
- [Prompt Engineering](https://www.youtube.com/watch?v=dOxUroR57xs)
- [Building AI Agents](https://www.youtube.com/watch?v=2xxziIWmaSA)

**Tools & Platforms:**
- [LangChain Hub](https://smith.langchain.com/) - Pre-built prompts and chains
- [LangSmith](https://smith.langchain.com/) - Debugging and monitoring
- [Hugging Face Spaces](https://huggingface.co/spaces) - Deploy agents

## 📝 Assignment

**Due Date**: End of Week 2

**Tasks:**
1. Build a conversational agent with memory
2. Integrate at least 2 external tools/APIs
3. Create a specialized agent for a specific domain
4. Document your agent's capabilities and limitations

**Submission Requirements:**
- Working code with proper documentation
- README explaining your agent's features
- Demo or screenshots of the agent in action
- Reflection on challenges and learnings

## 🎯 Next Week Preview

**Week 3: Fine-tuning Foundation Models**
- Understanding model fine-tuning
- LoRA and parameter-efficient methods
- Training data preparation
- Fine-tuning workflows and best practices

## 💡 Advanced Concepts

**Multi-Agent Systems:**
- Agent communication protocols
- Task delegation and coordination
- Conflict resolution strategies
- Scalability considerations

**Agent Evaluation:**
- Performance metrics
- Human evaluation methods
- Automated testing frameworks
- Continuous improvement strategies

## 🚀 Best Practices

**Agent Design:**
- Start simple and iterate
- Use appropriate memory for your use case
- Implement proper error handling
- Monitor agent performance and behavior

**Prompt Engineering:**
- Test prompts with diverse inputs
- Use temperature settings appropriately
- Implement prompt versioning
- Document prompt design decisions

**Security Considerations:**
- Validate all inputs
- Implement rate limiting
- Monitor for prompt injection attacks
- Secure API key management

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: The best agents are those that solve real problems. Focus on creating value for your users!* 