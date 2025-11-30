"""
Prompt templates and examples for SQL ReAct Agent
"""

SYSTEM_PROMPT_TEMPLATE = """You are a SQL Database Agent that helps users query a database using natural language.

## DATABASE SCHEMA:

___SCHEMA___

## AVAILABLE TOOLS:

___TOOLS___

## OUTPUT FORMAT:

CRITICAL: Each response must contain EXACTLY ONE of these:

Option 1 - Use a tool:
THOUGHT: your reasoning
ACTION: tool_name{"param": "value"}

Option 2 - Final answer:
FINAL ANSWER: your complete answer

RULES:
1. One output per turn: EITHER (THOUGHT + ACTION) OR (FINAL ANSWER)
2. NEVER generate OBSERVATION - system provides it
3. STOP after ACTION
4. ACTION format must be valid JSON: tool_name{"param": "value"}
5. Always THOUGHT first

___EXAMPLES___
"""

FEW_SHOT_EXAMPLES = """
## EXAMPLES:

Example 1: Simple Query

User: How many employees are in the database?

Agent:
THOUGHT: I can see the employees table. I'll count the rows.
ACTION: query_database{"query": "SELECT COUNT(*) as count FROM employees"}

OBSERVATION: Query returned 1 row(s):
count
-----
8

Agent:
FINAL ANSWER: There are 8 employees in the database.

---

Example 2: Multi-Step

User: How many people work in Engineering?

Agent:
THOUGHT: I need to find the department_id for Engineering first.
ACTION: query_database{"query": "SELECT id FROM departments WHERE name='Engineering'"}

OBSERVATION: Query returned 1 row(s):
id
--
1

Agent:
THOUGHT: Engineering has department_id=1. Now I'll count employees.
ACTION: query_database{"query": "SELECT COUNT(*) as count FROM employees WHERE department_id=1"}

OBSERVATION: Query returned 1 row(s):
count
-----
3  

Agent:
FINAL ANSWER: There are 3 people working in the Engineering department.

---

Remember: Each response is EITHER (THOUGHT + ACTION) OR (FINAL ANSWER). NEVER generate OBSERVATION.

NOW IT'S YOUR TURN.
"""


def build_tool_descriptions(tools: list) -> str:
    """Generate tool descriptions for prompt."""
    descriptions = []
    for i, tool in enumerate(tools, 1):
        descriptions.append(f"{i}. {tool.to_prompt_string()}")
    return "\n\n".join(descriptions)


def build_system_prompt(schema: str, tools: list, include_examples: bool = True) -> str:
    """Build complete system prompt."""
    tool_descriptions = build_tool_descriptions(tools)
    few_shot = FEW_SHOT_EXAMPLES if include_examples else ""
    
    # Simple string replacement - no .format() nonsense
    prompt = SYSTEM_PROMPT_TEMPLATE.replace("___SCHEMA___", schema)
    prompt = prompt.replace("___TOOLS___", tool_descriptions)
    prompt = prompt.replace("___EXAMPLES___", few_shot)
    
    return prompt


def extract_action_from_response(response: str) -> tuple[str, dict]:
    """Parse LLM response to extract tool name and parameters."""
    import re
    import json
    
    action_match = re.search(r'ACTION:\s*(\w+)', response)
    if not action_match:
        return None, None
    
    tool_name = action_match.group(1)
    
    start_pos = action_match.end()
    brace_start = response.find('{', start_pos)
    
    if brace_start == -1:
        return None, None
    
    # Find matching closing brace
    brace_count = 0
    brace_end = -1
    
    for i in range(brace_start, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    
    if brace_end == -1:
        return None, None
    
    json_str = response[brace_start:brace_end+1]
    
    try:
        params = json.loads(json_str)
        return tool_name, params
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON error: {str(e)}")
        print(f"[DEBUG] JSON was: {json_str}")
        return None, None


def has_final_answer(response: str) -> bool:
    """Check if response contains FINAL ANSWER."""
    return "FINAL ANSWER:" in response.upper()


def extract_final_answer(response: str) -> str:
    """Extract final answer from response."""
    import re
    
    pattern = r'FINAL ANSWER:\s*(.+)'
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""
