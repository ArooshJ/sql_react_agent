# ReAct Loop Internals - How Agents Actually Work

**Purpose:** Deep dive into how ReAct agents work under the hood, how we prevent hallucinated observations, and how this compares to framework implementations.

**Date:** 2025-11-30

---

## Table of Contents
1. [The Fundamental Question](#the-fundamental-question)
2. [How the ReAct Loop Actually Works](#how-the-react-loop-actually-works)
3. [Visual Flow Diagram](#visual-flow-diagram)
4. [The Code Implementation](#the-code-implementation)
5. [Conversation History Management](#conversation-history-management)
6. [The Hallucination Problem](#the-hallucination-problem)
7. [Four Strategies to Prevent Hallucinated Observations](#four-strategies-to-prevent-hallucinated-observations)
8. [How LangChain's invoke() Works](#how-langchains-invoke-works)
9. [Comparison: Manual vs bind_tools](#comparison-manual-vs-bind_tools)
10. [Key Insights](#key-insights)

---

## The Fundamental Question

**Question:** How does the LLM know to wait and receive the OBSERVATION after printing the ACTION block? How are we interrupting for the tool call?

**Answer:** The LLM **doesn't wait** - we interrupt it! This is the core mechanism of ReAct agents.

---

## How the ReAct Loop Actually Works

The LLM doesn't have any special "waiting" mechanism. Instead, we manually control a loop:

### Step-by-Step Process:

```
1. Send prompt to LLM (system + user query)
   ↓
2. LLM generates: THOUGHT + ACTION
   ↓
3. 🛑 WE STOP HERE - parse the output
   ↓
4. Extract tool name and parameters from ACTION
   ↓
5. Execute the Python function (our tool)
   ↓
6. Get result from tool
   ↓
7. Add OBSERVATION to conversation history (as "user" message)
   ↓
8. Call LLM again with updated history
   ↓
9. LLM sees OBSERVATION, generates next THOUGHT + ACTION or FINAL ANSWER
   ↓
10. Repeat until FINAL ANSWER
```

**Key Insight:** The LLM is called multiple times. Each call is independent - the LLM doesn't "remember" previous calls. All context comes from the conversation history we maintain.

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Call #1 (Initial)                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Input:                                                 │ │
│  │ - System: "You are a SQL agent..."                   │ │
│  │ - User: "How many employees?"                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Output:                                                │ │
│  │ THOUGHT: I need to query the employees table          │ │
│  │ ACTION: query_database{"query": "SELECT COUNT(*)..."}│ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  🛑 WE INTERRUPT HERE
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Python Agent Code (Our Loop)                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Parse: "ACTION: query_database{...}"               │ │
│  │ 2. Extract: tool_name = "query_database"              │ │
│  │            params = {"query": "SELECT COUNT(*)..."}   │ │
│  │ 3. Execute: result = tools["query_database"](params)  │ │
│  │ 4. Result: "Query returned 1 row(s): count: 8"        │ │
│  │ 5. Append to history:                                 │ │
│  │    - assistant: "THOUGHT... ACTION..."                │ │
│  │    - user: "OBSERVATION: Query returned..."           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM Call #2 (With Observation)                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Input (Full History):                                  │ │
│  │ - System: "You are a SQL agent..."                    │ │
│  │ - User: "How many employees?"                         │ │
│  │ - Assistant: "THOUGHT... ACTION..."                   │ │
│  │ - User: "OBSERVATION: Query returned... count: 8"     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Output:                                                │ │
│  │ FINAL ANSWER: There are 8 employees in the database.  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## The Code Implementation

### Minimal ReAct Loop Implementation

```python
class SQLReActAgent:
    def __init__(self, llm, tools, system_prompt):
        self.llm = llm
        self.tools_dict = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
    
    def run(self, user_query: str, max_steps: int = 10):
        """
        Main ReAct loop.
        
        Args:
            user_query: User's natural language question
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Final answer string
        """
        # Initialize conversation history
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Main loop - maximum max_steps iterations
        for step in range(max_steps):
            print(f"\n{'='*60}")
            print(f"STEP {step + 1}")
            print(f"{'='*60}")
            
            # ============================================================
            # CALL LLM
            # ============================================================
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=history,
                stop=["OBSERVATION:", "\nOBSERVATION"]  # Stop before hallucinating!
            )
            
            response_text = response.choices[0].message.content
            print(f"LLM Response:\n{response_text}")
            
            # ============================================================
            # ADD ASSISTANT MESSAGE TO HISTORY
            # ============================================================
            history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # ============================================================
            # CHECK: Is it done?
            # ============================================================
            if has_final_answer(response_text):
                final_answer = extract_final_answer(response_text)
                print(f"\n✓ Final Answer: {final_answer}")
                return final_answer
            
            # ============================================================
            # PARSE: Extract action and parameters
            # ============================================================
            action, params = extract_action_from_response(response_text)
            
            if not action:
                return "Error: No valid action found in response"
            
            print(f"\nParsed Action: {action}")
            print(f"Parameters: {params}")
            
            # ============================================================
            # EXECUTE: Run the tool
            # ============================================================
            if action not in self.tools_dict:
                return f"Error: Unknown tool '{action}'"
            
            tool_result = self.tools_dict[action](**params)
            print(f"\nTool Result:\n{tool_result}")
            
            # ============================================================
            # INJECT: Add observation to history (as "user" message)
            # ============================================================
            # This is the KEY: we pretend to be the user providing feedback!
            history.append({
                "role": "user",
                "content": f"OBSERVATION: {tool_result}"
            })
            
            # Loop continues → LLM gets called again with the observation
        
        return f"Error: Reached maximum steps ({max_steps})"
```

---

## Conversation History Management

### Example of History Evolution

**Initial State (Step 0):**
```python
history = [
    {
        "role": "system",
        "content": "You are a SQL Database Agent..."
    },
    {
        "role": "user",
        "content": "How many employees are in the database?"
    }
]
```

**After LLM Call #1:**
```python
history = [
    {"role": "system", "content": "You are a SQL Database Agent..."},
    {"role": "user", "content": "How many employees are in the database?"},
    {
        "role": "assistant",
        "content": "THOUGHT: I need to query the employees table\nACTION: query_database{\"query\": \"SELECT COUNT(*) FROM employees\"}"
    }
]
```

**After Tool Execution (We Add OBSERVATION):**
```python
history = [
    {"role": "system", "content": "You are a SQL Database Agent..."},
    {"role": "user", "content": "How many employees are in the database?"},
    {
        "role": "assistant",
        "content": "THOUGHT: I need to query the employees table\nACTION: query_database{\"query\": \"SELECT COUNT(*) FROM employees\"}"
    },
    {
        "role": "user",  # ← We act as the user!
        "content": "OBSERVATION: Query returned 1 row(s):\ncount\n-----\n8"
    }
]
```

**After LLM Call #2:**
```python
history = [
    {"role": "system", "content": "You are a SQL Database Agent..."},
    {"role": "user", "content": "How many employees are in the database?"},
    {"role": "assistant", "content": "THOUGHT... ACTION..."},
    {"role": "user", "content": "OBSERVATION: Query returned... count: 8"},
    {
        "role": "assistant",
        "content": "FINAL ANSWER: There are 8 employees in the database."
    }
]
```

**Key Point:** We manually construct the OBSERVATION message and inject it into the conversation. The LLM sees it as if the user provided it.

---

## The Hallucination Problem

### What Can Go Wrong?

Without proper safeguards, the LLM might generate:

```
THOUGHT: I need to query employees
ACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}
OBSERVATION: Query returned 1 row(s): count: 12    ← HALLUCINATED!
FINAL ANSWER: There are 12 employees.              ← Wrong answer!
```

**The problem:** LLM completes the entire conversation in one turn, including making up the OBSERVATION before we can execute the actual query!

### Why This Happens:

1. **LLMs are trained to complete patterns** - They've seen THOUGHT → ACTION → OBSERVATION → ANSWER patterns in training data
2. **No inherent "stopping" mechanism** - Without explicit instructions, they continue generating
3. **Eager completion** - LLMs want to be helpful and provide complete answers

### Real Example:

```python
# Without safeguards
response = llm.chat(history)
print(response.content)

# Output:
"""
THOUGHT: I'll query the database
ACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}
OBSERVATION: The query returned 15 rows
FINAL ANSWER: There are 15 employees
"""

# Meanwhile, actual database has 8 employees!
# The LLM never actually ran the query - it just guessed!
```

---

## Four Strategies to Prevent Hallucinated Observations

### Strategy 1: Stop Sequences ⭐ (Most Reliable)

**How it works:** Tell the LLM to stop generating text when it encounters specific tokens.

```python
response = llm.chat.completions.create(
    messages=history,
    stop=["OBSERVATION:", "\nOBSERVATION", "Observation:"]  # Stop tokens
)
```

**Result:**
```
Before (without stop):
"THOUGHT: Query employees\nACTION: query_database{...}\nOBSERVATION: fake data\nFINAL ANSWER: wrong"

After (with stop):
"THOUGHT: Query employees\nACTION: query_database{...}\n"
                                                        ↑
                                                    Stopped here!
```

**Why it works:** The LLM physically cannot continue past the stop sequence. The API interrupts generation immediately.

**Implementation:**
```python
# In agent.py
def _call_llm(self, history):
    response = self.llm.chat.completions.create(
        model=self.model_name,
        messages=history,
        temperature=0.0,  # Deterministic for reliability
        stop=["OBSERVATION:", "\nOBSERVATION", "Observation:"]
    )
    return response.choices[0].message.content
```

---

### Strategy 2: Careful Parsing + Validation

**How it works:** Even if the LLM generates OBSERVATION, we ignore everything after the ACTION line.

```python
def extract_action_from_response(response: str) -> tuple[str, dict]:
    """
    Parses LLM response to extract ONLY the first ACTION.
    Ignores anything that comes after.
    """
    import re
    import json
    
    # Find ACTION line
    lines = response.split('\n')
    action_line = None
    
    for line in lines:
        if line.strip().startswith('ACTION:'):
            action_line = line
            break  # Stop at first ACTION, ignore rest
    
    if not action_line:
        return None, None
    
    # Pattern: ACTION: tool_name{json}
    pattern = r'ACTION:\s*(\w+)\s*(\{[^}]*\})'
    match = re.search(pattern, action_line)
    
    if not match:
        return None, None
    
    tool_name = match.group(1)
    json_str = match.group(2)
    
    try:
        params = json.loads(json_str)
        return tool_name, params
    except json.JSONDecodeError:
        return None, None
```

**Example:**
```python
response = """
THOUGHT: Query employees
ACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}
OBSERVATION: Fake data here
FINAL ANSWER: Wrong answer
"""

action, params = extract_action_from_response(response)
# Returns: ("query_database", {"query": "SELECT COUNT(*) FROM employees"})
# Ignores the fake OBSERVATION and FINAL ANSWER!
```

---

### Strategy 3: Few-Shot Examples in Prompt

**How it works:** Show the LLM correct examples where OBSERVATION is provided by the system.

```python
FEW_SHOT_EXAMPLES = """
EXAMPLES:

User: How many employees?

THOUGHT: I need to query the employees table
ACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}
OBSERVATION: Query returned 1 row(s):  ← System provides this!
count
-----
8

FINAL ANSWER: There are 8 employees.
"""
```

**Why it works:** The LLM learns the pattern:
- "I generate THOUGHT and ACTION"
- "The system provides OBSERVATION"
- "I respond to the OBSERVATION"

**Teaching effect:** The LLM internalizes that OBSERVATION comes from external source, not self-generated.

---

### Strategy 4: Explicit Instructions in System Prompt

**How it works:** Directly tell the LLM not to generate observations.

```python
SYSTEM_PROMPT = """
You are a SQL Database Agent.

OUTPUT FORMAT:
THOUGHT: <your reasoning>
ACTION: <tool_name>{<json_params>}

CRITICAL RULES:
1. Generate ONLY ONE ACTION per turn
2. After ACTION, STOP generating
3. DO NOT generate OBSERVATION yourself
4. The system will provide OBSERVATION to you
5. Wait for OBSERVATION before continuing
6. Only generate FINAL ANSWER when you have real data
"""
```

**Why it works:** Explicit instructions prime the LLM's behavior. Modern LLMs are very good at following clear instructions.

---

### Combining All Four Strategies

```python
class SQLReActAgent:
    def __init__(self, llm, tools, schema):
        self.llm = llm
        self.tools_dict = {tool.name: tool for tool in tools}
        
        # Strategy 3: Use prompts with few-shot examples
        self.system_prompt = build_system_prompt(
            schema=schema,
            tools=tools,
            include_examples=True  # ← Few-shot examples included
        )
    
    def _call_llm(self, history):
        # Strategy 1: Use stop sequences
        response = self.llm.chat.completions.create(
            messages=history,
            stop=["OBSERVATION:", "\nOBSERVATION"]  # ← Stop tokens
        )
        return response.choices[0].message.content
    
    def run(self, user_query: str):
        history = [
            # Strategy 4: Explicit instructions in system prompt
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        for step in range(10):
            response = self._call_llm(history)
            history.append({"role": "assistant", "content": response})
            
            if has_final_answer(response):
                return extract_final_answer(response)
            
            # Strategy 2: Careful parsing (ignore anything after ACTION)
            action, params = extract_action_from_response(response)
            
            if not action:
                return "Error: No valid action"
            
            # Execute tool and inject OBSERVATION
            result = self.tools_dict[action](**params)
            history.append({
                "role": "user",
                "content": f"OBSERVATION: {result}"
            })
```

---

## How LangChain's `invoke()` Works

### LangChain AgentExecutor

```python
from langchain.agents import AgentExecutor, create_react_agent

# Setup
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run
result = agent_executor.invoke({"input": "How many employees?"})
```

### What `invoke()` Does Internally

**Simplified implementation:**

```python
class AgentExecutor:
    def invoke(self, input_dict):
        # Initialize
        messages = [{"role": "user", "content": input_dict["input"]}]
        max_iterations = 10
        
        for i in range(max_iterations):
            # Step 1: Agent plans next action
            agent_output = self.agent.plan(
                intermediate_steps=messages,
                **input_dict
            )
            
            # Step 2: Check if done
            if agent_output.return_values:
                return {"output": agent_output.return_values}
            
            # Step 3: Execute tool
            tool_name = agent_output.tool
            tool_input = agent_output.tool_input
            
            tool_output = self.tools[tool_name].run(tool_input)
            
            # Step 4: Add to history
            messages.append({
                "role": "assistant",
                "content": agent_output.log  # The ACTION
            })
            messages.append({
                "role": "function",  # or "user" depending on API
                "name": tool_name,
                "content": tool_output  # The OBSERVATION
            })
        
        raise ValueError("Agent stopped due to max iterations")
```

**Key similarities to our implementation:**
1. Loop with max iterations
2. Call agent/LLM to get action
3. Execute tool
4. Add observation to history
5. Repeat until done

**The difference:** LangChain abstracts it away, we implement it explicitly.

---

## Comparison: Manual vs bind_tools

### Approach 1: Our Manual Implementation

```python
# We define tools in the prompt (text-based)
system_prompt = """
AVAILABLE TOOLS:
1. query_database(query: str): Executes SELECT query
2. list_tables(): Lists all tables
"""

# Regular chat API call
response = llm.chat.completions.create(
    messages=history,
    stop=["OBSERVATION:"]  # We add stop sequences
)

# We parse the text response
response_text = response.choices[0].message.content
action, params = extract_action_from_response(response_text)

# We execute the tool
result = tools[action](**params)

# We manually add OBSERVATION
history.append({"role": "user", "content": f"OBSERVATION: {result}"})
```

**Pros:**
- ✅ Works with ANY LLM (Groq, Gemini, local models)
- ✅ Full control over parsing and execution
- ✅ Debuggable - see exactly what's happening
- ✅ No framework dependencies

**Cons:**
- ❌ Manual parsing (more code)
- ❌ Need to handle stop sequences
- ❌ Possible hallucinations (need safeguards)

---

### Approach 2: Native Tool Calling (bind_tools)

```python
from langchain_openai import ChatOpenAI

# Define tools with schema
tools = [
    Tool(
        name="query_database",
        func=query_database,
        description="Executes SQL query",
        args_schema=QueryInput  # Pydantic schema
    )
]

# Bind tools to LLM
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)

# LLM response includes structured tool calls
response = llm_with_tools.invoke(history)

# Response format:
{
    "role": "assistant",
    "content": null,  # No text content!
    "tool_calls": [
        {
            "id": "call_123",
            "function": {
                "name": "query_database",
                "arguments": '{"query": "SELECT COUNT(*) FROM employees"}'
            }
        }
    ]
}

# Execute tool
tool_output = tools["query_database"].run(
    json.loads(response.tool_calls[0].function.arguments)
)

# Add tool result to history (special format)
history.append({
    "role": "tool",
    "tool_call_id": response.tool_calls[0].id,
    "content": tool_output
})
```

**Pros:**
- ✅ No hallucinated observations (API format prevents it)
- ✅ Structured output (JSON schema enforced)
- ✅ Less parsing code
- ✅ Automatic validation

**Cons:**
- ❌ Only works with OpenAI/Anthropic (limited LLM support)
- ❌ Less control (framework magic)
- ❌ Harder to debug
- ❌ Locked into specific APIs

---

### Side-by-Side Comparison

| Aspect | Manual (Our Approach) | Native (bind_tools) |
|--------|----------------------|---------------------|
| **LLM Support** | Any LLM | OpenAI, Anthropic only |
| **Tool Format** | Text in prompt | JSON schema |
| **Parsing** | Manual regex/JSON | Automatic |
| **Hallucinations** | Possible (need safeguards) | Prevented by API |
| **Control** | Full | Limited |
| **Debugging** | Easy (see everything) | Harder (abstracted) |
| **Code Complexity** | Higher | Lower |
| **Assignment Compliance** | ✅ Allowed | ❌ Framework prohibited |
| **Learning Value** | High (see internals) | Low (black box) |

---

## Key Insights

### 1. The LLM Never "Waits"

The LLM doesn't have any concept of waiting or persistence:
- Each `llm.chat()` call is independent
- The LLM doesn't remember previous calls
- All "memory" comes from the conversation history we maintain

### 2. We Control the Loop

The agent loop is just Python code:
```python
while not done:
    response = call_llm()
    if has_final_answer(response):
        return answer
    action = parse_action(response)
    result = execute_tool(action)
    add_observation_to_history(result)
```

### 3. Conversation History Is Everything

The history array is the ONLY way the LLM knows what happened:
```python
history = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": question},
    {"role": "assistant", "content": thought_and_action},
    {"role": "user", "content": observation},  # ← We add this!
    ...
]
```

### 4. Hallucinations Are a Real Risk

Without safeguards, the LLM will happily:
- Generate fake observations
- Make up data
- Provide confident but wrong answers

**Solution:** Use all 4 strategies:
1. Stop sequences
2. Careful parsing
3. Few-shot examples
4. Explicit instructions

### 5. Frameworks Just Wrap This Pattern

LangChain's `invoke()`, LangGraph's nodes, AutoGPT, etc. - they all implement the same basic pattern we're building. The difference is abstraction level, not fundamental mechanism.

### 6. Manual Implementation = Deep Understanding

By building it from scratch, you learn:
- How prompts control behavior
- How parsing works
- How to debug when things go wrong
- How to extend for custom use cases

---

## Conclusion

The ReAct loop is surprisingly simple:
1. Call LLM
2. Parse ACTION
3. Execute tool
4. Add OBSERVATION to history
5. Repeat

The complexity comes from:
- Preventing hallucinations (stop sequences!)
- Robust parsing (handle edge cases)
- Error handling (what if tool fails?)
- Managing conversation state

**Our implementation gives you full control and understanding of every step.**

---

## Next Steps

Now that we understand the internals, we'll implement `agent.py` with:
- ✅ Proper stop sequences
- ✅ Robust parsing
- ✅ Clear history management
- ✅ Error handling
- ✅ All 4 hallucination prevention strategies

Let's build it! 🚀
