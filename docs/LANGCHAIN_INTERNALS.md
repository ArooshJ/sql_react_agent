# LangChain Invoke and Tool Calls - Deep Dive

**Purpose:** Detailed exploration of how LangChain's `invoke()`, `bind_tools()`, and agent execution actually work under the hood, including the role of `.plan()` and the exact sequence of LLM calls.

**Date:** 2025-11-30

---

## Table of Contents
1. [The Question](#the-question)
2. [LangChain Agent Architecture](#langchain-agent-architecture)
3. [The plan() Method - What It Actually Does](#the-plan-method---what-it-actually-does)
4. [Complete invoke() Flow with LLM Call Tracking](#complete-invoke-flow-with-llm-call-tracking)
5. [bind_tools() Under the Hood](#bind_tools-under-the-hood)
6. [OpenAI Function Calling - The Native API](#openai-function-calling---the-native-api)
7. [ReAct Agent vs Function Calling Agent](#react-agent-vs-function-calling-agent)
8. [Real Code Examples from LangChain Source](#real-code-examples-from-langchain-source)
9. [Actual Network Traffic](#actual-network-traffic)
10. [Performance Implications](#performance-implications)

---

## The Question

**Q: Is `.plan()` a separate LLM invocation compared to the tool call?**

**A: YES!** And in fact, for a single tool execution cycle, there can be:
- **1 LLM call** in `.plan()` to decide what action to take
- **1 Python function call** to execute the tool
- **1 more LLM call** in the next `.plan()` to process the result

Let's trace through exactly what happens.

---

## LangChain Agent Architecture

### Core Components

```python
from langchain.agents import AgentExecutor, create_react_agent

# Component 1: The Agent (decision maker)
agent = create_react_agent(
    llm=llm,           # Language model
    tools=tools,       # Available tools
    prompt=prompt      # ReAct prompt template
)

# Component 2: The Executor (orchestrator)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run
result = agent_executor.invoke({"input": "How many employees?"})
```

### Key Classes

```
AgentExecutor (orchestrator)
    ├── Agent (decision logic)
    │   ├── LLM (language model)
    │   ├── Prompt (instructions)
    │   └── .plan() method ← This calls the LLM!
    └── Tools (actions)
```

---

## The plan() Method - What It Actually Does

### Method Signature

```python
class Agent:
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """
        Decides next action based on current state.
        
        This method CALLS THE LLM to determine what to do next.
        
        Args:
            intermediate_steps: History of (action, observation) pairs
            **kwargs: Additional inputs (user query, etc.)
            
        Returns:
            AgentAction (with tool name + input) or AgentFinish (final answer)
        """
        # Build prompt from intermediate steps
        full_prompt = self._construct_prompt(intermediate_steps, **kwargs)
        
        # 🔥 CRITICAL: This is an LLM call!
        llm_output = self.llm(full_prompt)
        
        # Parse LLM output
        parsed = self.output_parser.parse(llm_output)
        
        return parsed  # AgentAction or AgentFinish
```

### What Happens Inside plan()

```
1. Builds prompt from conversation history
   ↓
2. 🌐 CALLS LLM (actual API request)
   ↓
3. Gets response (text or structured output)
   ↓
4. Parses response to extract:
   - Tool name
   - Tool input
   - Or final answer
   ↓
5. Returns AgentAction or AgentFinish object
```

**Key Point:** Every call to `.plan()` = 1 LLM API call

---

## Complete invoke() Flow with LLM Call Tracking

### Real AgentExecutor Code (Simplified from LangChain Source)

```python
class AgentExecutor:
    def invoke(self, inputs: Dict) -> Dict:
        """
        Main execution loop.
        Tracks LLM calls and tool executions.
        """
        # Initialize
        intermediate_steps = []  # History of (action, observation)
        iterations = 0
        max_iterations = 15
        
        # Track LLM calls
        llm_call_count = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # ================================================================
            # STEP 1: CALL .plan() → This calls the LLM!
            # ================================================================
            print(f"\n[Iteration {iterations}]")
            print("Calling agent.plan()...")
            
            llm_call_count += 1  # ← LLM Call #N
            
            agent_decision = self.agent.plan(
                intermediate_steps=intermediate_steps,
                **inputs
            )
            
            print(f"✓ LLM call #{llm_call_count} completed")
            print(f"  Decision type: {type(agent_decision).__name__}")
            
            # ================================================================
            # STEP 2: CHECK - Is agent done?
            # ================================================================
            if isinstance(agent_decision, AgentFinish):
                print("Agent returned AgentFinish")
                print(f"Total LLM calls: {llm_call_count}")
                return self._return(
                    agent_decision.return_values,
                    intermediate_steps
                )
            
            # ================================================================
            # STEP 3: EXECUTE TOOL (Python function, not LLM call)
            # ================================================================
            print(f"Executing tool: {agent_decision.tool}")
            
            tool = self.tools_by_name[agent_decision.tool]
            
            # This is just a Python function call
            tool_output = tool.run(agent_decision.tool_input)
            
            print(f"✓ Tool execution completed")
            
            # ================================================================
            # STEP 4: SAVE TO HISTORY
            # ================================================================
            intermediate_steps.append((agent_decision, tool_output))
            
            # Loop continues → plan() gets called again → another LLM call
        
        # Max iterations reached
        print(f"Max iterations reached. Total LLM calls: {llm_call_count}")
        return self._return({"output": "Agent stopped"}, intermediate_steps)
```

### Example Execution Trace

```
User Query: "How many employees in Engineering department?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Iteration 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calling agent.plan()...
  📡 LLM API Call #1
  Request: {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a ReAct agent..."},
      {"role": "user", "content": "How many employees in Engineering?"}
    ]
  }
  Response: "THOUGHT: I need to find Engineering dept ID\nACTION: query_database..."
  
✓ LLM call #1 completed
  Decision type: AgentAction
  Tool: query_database
  Input: {"query": "SELECT id FROM departments WHERE name='Engineering'"}

Executing tool: query_database
  🐍 Python Function Call (NOT an LLM call)
  Running: query_database(query="SELECT id FROM...")
  Result: "id\n--\n1"
  
✓ Tool execution completed

Intermediate steps: [(AgentAction(...), "id: 1")]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Iteration 2]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calling agent.plan()...
  📡 LLM API Call #2
  Request: {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a ReAct agent..."},
      {"role": "user", "content": "How many employees in Engineering?"},
      {"role": "assistant", "content": "THOUGHT... ACTION..."},
      {"role": "function", "name": "query_database", "content": "id: 1"}
    ]
  }
  Response: "THOUGHT: Now I know Engineering ID is 1, query employees\nACTION: query_database..."
  
✓ LLM call #2 completed
  Decision type: AgentAction
  Tool: query_database
  Input: {"query": "SELECT COUNT(*) FROM employees WHERE department_id=1"}

Executing tool: query_database
  🐍 Python Function Call
  Running: query_database(query="SELECT COUNT(*)...")
  Result: "count\n-----\n3"
  
✓ Tool execution completed

Intermediate steps: [
  (AgentAction(...), "id: 1"),
  (AgentAction(...), "count: 3")
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Iteration 3]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calling agent.plan()...
  📡 LLM API Call #3
  Request: {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a ReAct agent..."},
      {"role": "user", "content": "How many employees in Engineering?"},
      {"role": "assistant", "content": "THOUGHT... ACTION..."},
      {"role": "function", "content": "id: 1"},
      {"role": "assistant", "content": "THOUGHT... ACTION..."},
      {"role": "function", "content": "count: 3"}
    ]
  }
  Response: "THOUGHT: I have the answer\nFINAL ANSWER: There are 3 employees..."
  
✓ LLM call #3 completed
  Decision type: AgentFinish
  Return values: {"output": "There are 3 employees in Engineering"}

Agent returned AgentFinish
Total LLM calls: 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final Result: "There are 3 employees in Engineering department"
```

### Key Observations

**For this single user query:**
- **3 LLM calls** (via `.plan()`)
- **2 tool executions** (Python functions)
- **3 iterations** total

**Pattern:**
```
Iteration N:
  plan() → LLM call → AgentAction
  tool.run() → Python function → observation
  
Iteration N+1:
  plan() → LLM call (with previous observation) → ...
```

---

## bind_tools() Under the Hood

### What bind_tools() Actually Does

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)
```

**This is just syntactic sugar!** It modifies the LLM wrapper to automatically:

1. Convert LangChain tools to OpenAI function format
2. Include `tools` parameter in every API call
3. Parse tool calls from responses

### Before bind_tools (manual):

```python
# Define tool in OpenAI format
tool_definition = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Executes SQL query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query"}
            },
            "required": ["query"]
        }
    }
}

# Manual API call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[tool_definition],  # ← Manually add
    tool_choice="auto"
)

# Manual parsing
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute tool...
```

### After bind_tools (automatic):

```python
llm_with_tools = llm.bind_tools([langchain_tool])

# LangChain does the above automatically
response = llm_with_tools.invoke(messages)

# Tool calls already parsed
if hasattr(response, 'tool_calls'):
    tool_call = response.tool_calls[0]
    # ...
```

**Under the hood, bind_tools creates a wrapper:**

```python
class ToolBindingWrapper:
    def __init__(self, base_llm, tools):
        self.base_llm = base_llm
        self.tool_definitions = self._convert_tools(tools)
    
    def _convert_tools(self, langchain_tools):
        """Convert LangChain tools to OpenAI function format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema()  # Pydantic → JSON
                }
            }
            for tool in langchain_tools
        ]
    
    def invoke(self, messages):
        # Add tools to every call
        response = self.base_llm.invoke(
            messages,
            tools=self.tool_definitions,  # ← Automatically added
            tool_choice="auto"
        )
        # Parse and structure response
        return self._parse_response(response)
```

---

## OpenAI Function Calling - The Native API

### How OpenAI's Native Tool Calling Works

When you use OpenAI's function calling (which `bind_tools` uses):

```python
import openai

# Define function/tool
functions = [{
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Executes SQL query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]

# Call with tools
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "How many employees?"}
    ],
    tools=functions,
    tool_choice="auto"  # Let model decide
)
```

### Response Format

```python
# OpenAI returns structured tool calls
{
    "id": "chatcmpl-123",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": null,  # ← No text when tool calling!
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "arguments": '{"query": "SELECT COUNT(*) FROM employees"}'
                    }
                }
            ]
        },
        "finish_reason": "tool_calls"
    }]
}
```

**Key Points:**
- `content` is null (no text output)
- `tool_calls` contains structured function call
- `arguments` is a JSON string (need to parse)

### Executing and Continuing

```python
# Step 1: Extract tool call
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_args = json.loads(tool_call.function.arguments)

# Step 2: Execute function (Python)
result = my_functions[function_name](**function_args)

# Step 3: Add to conversation
messages.append(response.choices[0].message)  # Assistant's tool call
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": str(result)
})

# Step 4: Call again with result
final_response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,  # Now includes tool result
    tools=functions
)

# Now model uses the result to generate final answer
```

### Complete Flow

```
User: "How many employees?"

Call #1:
  Request: messages + tools definitions
  Response: tool_call(query_database, {"query": "SELECT COUNT(*)"})
  
Execute tool: → "count: 8"

Call #2:
  Request: messages + tool result
  Response: "There are 8 employees"
```

**Still 2 LLM calls!**
1. To decide to call tool
2. To use tool result

---

## ReAct Agent vs Function Calling Agent

### Approach 1: ReAct Agent (Text-based)

```python
# Prompt-based tool calling
agent = create_react_agent(llm, tools, prompt)

# Prompt includes:
"""
Available tools:
- query_database(query: str): Executes SQL

Format:
THOUGHT: <reasoning>
ACTION: tool_name{json_args}
"""

# LLM Call #1:
Input: User query + prompt
Output: "THOUGHT: Need to query\nACTION: query_database{...}"

# Parse text, execute tool

# LLM Call #2:
Input: Previous + "OBSERVATION: result"
Output: "FINAL ANSWER: ..."
```

**Characteristics:**
- ✅ Works with any LLM
- ✅ Visible reasoning (THOUGHT)
- ❌ Text parsing (fragile)
- ❌ Possible hallucinations
- ❌ Needs stop sequences

### Approach 2: Function Calling Agent (Native API)

```python
# Native tool support
agent = create_openai_functions_agent(llm, tools)

# LLM Call #1:
Input: User query + tool definitions (JSON schema)
Output: Structured tool_call object
{
  "tool_calls": [{
    "function": {"name": "query_database", "arguments": "{...}"}
  }]
}

# Execute tool

# LLM Call #2:
Input: Previous + tool result (structured)
Output: "There are 8 employees"
```

**Characteristics:**
- ✅ Structured output (no parsing)
- ✅ No hallucinations
- ✅ Cleaner code
- ❌ Only OpenAI/Anthropic
- ❌ Less transparent (no THOUGHT)
- ❌ Locked to specific APIs

### Side-by-Side

| Aspect | ReAct | Function Calling |
|--------|-------|------------------|
| **LLM Support** | Any | OpenAI, Anthropic, some others |
| **Output Format** | Text (needs parsing) | Structured JSON |
| **Tool Call** | Text: `ACTION: tool{...}` | Object: `tool_calls[0]` |
| **Reasoning** | Visible (THOUGHT) | Hidden |
| **Reliability** | Moderate (text parsing) | High (schema enforced) |
| **Hallucinations** | Possible | Prevented |
| **Transparency** | High | Low |
| **Debugging** | Easy (see THOUGHT) | Harder |

---

## Real Code Examples from LangChain Source

### AgentExecutor.invoke() - Actual LangChain Code

```python
# From langchain/agents/agent.py

class AgentExecutor(Chain):
    """Execute an agent."""
    
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
    max_iterations: Optional[int] = 15
    
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        
        # Step 1: Prepare
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        iterations = 0
        
        # Step 2: Loop
        while self._should_continue(iterations, time_elapsed):
            # ===== CRITICAL: This is an LLM call! =====
            next_step_output = self._take_next_step(
                name_to_tool_map,
                intermediate_steps,
                run_manager=run_manager,
            )
            
            # Check if done
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )
            
            # Add to history
            intermediate_steps.extend(next_step_output)
            iterations += 1
        
        # Stopped due to iteration limit
        return self._return(
            AgentFinish({"output": "Agent stopped"}, ""),
            intermediate_steps,
        )
    
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the agent loop."""
        
        # ===== HERE'S THE LLM CALL =====
        output = self.agent.plan(
            intermediate_steps,
            **inputs,
            **run_manager.metadata if run_manager else {},
        )
        
        # If finished, return
        if isinstance(output, AgentFinish):
            return output
        
        # Execute tool
        actions = output if isinstance(output, list) else [output]
        result = []
        
        for agent_action in actions:
            # Get tool
            tool = name_to_tool_map[agent_action.tool]
            
            # Execute (Python function call, NOT LLM)
            observation = tool.run(
                agent_action.tool_input,
                run_manager=run_manager,
            )
            
            result.append((agent_action, observation))
        
        return result
```

### Agent.plan() - Where the LLM Call Happens

```python
# From langchain/agents/react/agent.py

class ReActAgent(Agent):
    """React agent that uses text-based tool calling."""
    
    @property
    def llm_chain(self) -> LLMChain:
        return self._llm_chain
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do."""
        
        # Build full input
        full_inputs = {
            "intermediate_steps": intermediate_steps,
            **kwargs,
        }
        
        # ===== THIS IS THE ACTUAL LLM CALL =====
        # llm_chain internally calls: llm.invoke(prompt)
        output = self.llm_chain.predict(**full_inputs)
        
        # Parse output using output parser
        return self.output_parser.parse(output)
```

### LLMChain.predict() - The Actual API Call

```python
# From langchain/chains/llm.py

class LLMChain(Chain):
    """Chain to run queries against LLMs."""
    
    llm: BaseLanguageModel
    prompt: BasePromptTemplate
    
    def predict(self, **kwargs: Any) -> str:
        """Format prompt and call LLM."""
        
        # Format the prompt
        formatted_prompt = self.prompt.format(**kwargs)
        
        # ===== ACTUAL LLM API CALL =====
        response = self.llm.invoke(formatted_prompt)
        
        return response
```

So the call chain is:
```
agent_executor.invoke()
  → _take_next_step()
    → agent.plan()
      → llm_chain.predict()
        → llm.invoke() ← ACTUAL API CALL HERE
```

---

## Actual Network Traffic

### What Goes Over the Wire

Using a network sniffer or OpenAI's logging, here's what actually happens:

**Request #1 (First plan() call):**
```http
POST https://api.openai.com/v1/chat/completions
Content-Type: application/json

{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a ReAct agent. Use tools to answer questions.\n\nTools:\n- query_database(query: str)"
    },
    {
      "role": "user",
      "content": "How many employees?"
    }
  ],
  "temperature": 0,
  "stop": ["OBSERVATION:", "\nObservation:"]
}
```

**Response #1:**
```json
{
  "id": "chatcmpl-abc123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "THOUGHT: I need to query the database\nACTION: query_database{\"query\": \"SELECT COUNT(*) FROM employees\"}"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 234,
    "completion_tokens": 45,
    "total_tokens": 279
  }
}
```

**[Tool execution happens locally - no network call]**

**Request #2 (Second plan() call with observation):**
```http
POST https://api.openai.com/v1/chat/completions

{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "How many employees?"},
    {
      "role": "assistant",
      "content": "THOUGHT: I need to query...\nACTION: query_database{...}"
    },
    {
      "role": "user",
      "content": "OBSERVATION: Query returned 1 row(s):\ncount\n-----\n8"
    }
  ],
  "temperature": 0,
  "stop": ["OBSERVATION:", "\nObservation:"]
}
```

**Response #2:**
```json
{
  "id": "chatcmpl-def456",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "FINAL ANSWER: There are 8 employees in the database."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 312,
    "completion_tokens": 18,
    "total_tokens": 330
  }
}
```

**Total network calls: 2 HTTP requests to OpenAI**

---

## Performance Implications

### Cost Analysis

**Scenario:** User asks "How many employees in Engineering?"

**ReAct Agent (3 iterations):**
```
LLM Call #1: 500 input tokens → 50 output tokens
LLM Call #2: 600 input tokens → 60 output tokens  
LLM Call #3: 750 input tokens → 40 output tokens

Total: 1,850 input + 150 output = 2,000 tokens
Cost (GPT-4): ~$0.12
Time: ~6 seconds (3 × 2s per call)
```

**Function Calling Agent (2 iterations):**
```
LLM Call #1: 450 input tokens → 30 output tokens (structured)
LLM Call #2: 550 input tokens → 25 output tokens

Total: 1,000 input + 55 output = 1,055 tokens
Cost (GPT-4): ~$0.06
Time: ~4 seconds (2 × 2s per call)
```

**Direct (no agent):**
```
Single LLM call with schema in context:
800 input tokens → 50 output tokens

Total: 850 tokens
Cost: ~$0.05
Time: ~2 seconds
```

### Optimization Strategies

1. **Reduce iterations:** Better prompts → fewer back-and-forth
2. **Cache schema:** Don't repeat schema in every call
3. **Use cheaper models:** GPT-3.5 for simple queries
4. **Parallel tool calls:** Execute multiple tools at once (OpenAI supports this)
5. **Skip unnecessary planning:** If query is simple, go straight to tool

---

## Summary

### Key Takeaways

**1. `.plan()` IS an LLM call**
- Every call to `agent.plan()` = 1 API request to OpenAI/Anthropic/etc.
- It's not a heuristic or rule-based system
- It's literally asking the LLM "what should I do next?"

**2. invoke() orchestrates multiple LLM calls**
```
User query → plan() → LLM Call #1 → tool execution → 
             plan() → LLM Call #2 → tool execution → 
             plan() → LLM Call #3 → final answer
```

**3. Tool execution is NOT an LLM call**
- Tools are just Python functions
- They run locally (or call external APIs)
- No LLM involved in executing tools

**4. bind_tools is syntactic sugar**
- Converts LangChain tools to API format
- Automatically includes tool definitions in calls
- Parses structured responses
- Still makes the same API calls under the hood

**5. Different agent types = different API usage**
- ReAct: Text-based, works anywhere, more calls
- Function Calling: Structured, OpenAI-only, fewer calls
- Our manual approach: Most control, most code

### The Complete Picture

```
User → AgentExecutor.invoke()
          ↓
       [LOOP: max 15 iterations]
          ↓
       ┌──────────────────┐
       │  agent.plan()    │ ← LLM Call #N
       │  (LLM API call)  │
       └────────┬─────────┘
                ↓
         [Parse response]
                ↓
    ┌───────────┴────────────┐
    │                        │
  Done?                  Need tool?
    │                        │
    ↓                        ↓
Return              Execute tool
result              (Python function)
                           ↓
                    Add to history
                           ↓
                    [Loop continues]
```

**Every iteration = 1 LLM call + 0-1 tool executions**

---

This deep dive shows why building from scratch gives you better understanding - you see every LLM call, every tool execution, and every decision point clearly!
