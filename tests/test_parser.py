"""
Test the fixed ACTION parser
"""

def extract_action_from_response_fixed(response: str):
    """
    Parses LLM response to extract tool name and parameters.
    Handles multi-line JSON and nested braces correctly.
    """
    import re
    import json
    
    # First, find "ACTION:" in the response
    action_match = re.search(r'ACTION:\s*(\w+)', response)
    if not action_match:
        return None, None
    
    tool_name = action_match.group(1)
    
    # Find the opening brace after the tool name
    start_pos = action_match.end()
    brace_start = response.find('{', start_pos)
    
    if brace_start == -1:
        return None, None
    
    # Find matching closing brace (handle nested braces)
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
    
    # Extract JSON string
    json_str = response[brace_start:brace_end+1]
    
    try:
        params = json.loads(json_str)
        return tool_name, params
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"JSON string was: {json_str}")
        return None, None


# Test cases
test_responses = [
    # Simple query
    'THOUGHT: I need to count.\nACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}',
    
    # Multi-line query (the problematic case)
    '''THOUGHT: To find out which department pays the maximum average salary to its employees, I need to join employees and departments tables.
ACTION: query_database{"query": "SELECT d.name, AVG(e.salary) as avg_salary FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name ORDER BY avg_salary DESC LIMIT 1"}''',
    
    # Empty params
    'THOUGHT: Let me list tables\nACTION: list_tables{}',
]

print("Testing ACTION extraction:\n")
for i, test in enumerate(test_responses, 1):
    print(f"Test {i}:")
    print(f"Input: {test[:80]}...")
    tool, params = extract_action_from_response_fixed(test)
    print(f"Result: tool={tool}, params={params}")
    print()
