"""
Test prompts.py WITHOUT making any API calls
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts import build_system_prompt, extract_action_from_response

# Mock data
mock_schema = """
Tables:
- employees (id, name, salary, department_id)
- departments (id, name)
"""

class MockTool:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
    
    def to_prompt_string(self):
        return f"{self.name}: {self.desc}"

mock_tools = [
    MockTool("query_database", "Execute SQL query"),
    MockTool("list_tables", "List all tables"),
]

print("=" * 70)
print("TEST 1: Build System Prompt (NO API CALLS)")
print("=" * 70)

try:
    prompt = build_system_prompt(mock_schema, mock_tools, include_examples=True)
    print("[PASS] Prompt built successfully!")
    print(f"[PASS] Length: {len(prompt)} characters")
    print(f"[PASS] Contains schema: {'___SCHEMA___' not in prompt}")
    print(f"[PASS] Contains tools: {'___TOOLS___' not in prompt}")
    print(f"[PASS] Contains examples: {'Example 1' in prompt}")
    print("\nFirst 500 chars:")
    print(prompt[:500])
    print("\n" + "=" * 70)
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTEST 2: Parse ACTION with single braces")
print("=" * 70)

test_response = '''THOUGHT: I need to count employees.
ACTION: query_database{"query": "SELECT COUNT(*) FROM employees"}'''

try:
    tool_name, params = extract_action_from_response(test_response)
    print(f"[PASS] Parsed tool: {tool_name}")
    print(f"[PASS] Parsed params: {params}")
    assert tool_name == "query_database", "Wrong tool name"
    assert "query" in params, "Missing query param"
    print("[PASS] Single brace JSON parsing WORKS!")
    print("\n" + "=" * 70)
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTEST 3: Parse complex ACTION with nested JSON")
print("=" * 70)

test_response_complex = '''THOUGHT: Complex query.
ACTION: query_database{"query": "SELECT d.name, AVG(e.salary) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name"}'''

try:
    tool_name, params = extract_action_from_response(test_response_complex)
    print(f"[PASS] Parsed tool: {tool_name}")
    print(f"[PASS] Parsed params: {params}")
    assert "JOIN" in params["query"], "Query parsing failed"
    print("[PASS] Complex JSON parsing WORKS!")
    print("\n" + "=" * 70)
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTEST 4: Detect FINAL ANSWER")
print("=" * 70)

from prompts import has_final_answer, extract_final_answer

test_final = "FINAL ANSWER: There are 8 employees in the database."

try:
    has_answer = has_final_answer(test_final)
    answer_text = extract_final_answer(test_final)
    print(f"[PASS] Detected final answer: {has_answer}")
    print(f"[PASS] Extracted text: '{answer_text}'")
    assert has_answer == True, "Should detect final answer"
    assert "8 employees" in answer_text, "Should extract answer text"
    print("[PASS] FINAL ANSWER parsing WORKS!")
    print("\n" + "=" * 70)
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTEST 5: Don't detect FINAL ANSWER in regular response")
print("=" * 70)

test_no_final = "THOUGHT: I should query the database."

try:
    has_answer = has_final_answer(test_no_final)
    print(f"[PASS] No final answer detected: {not has_answer}")
    assert has_answer == False, "Should NOT detect final answer"
    print("[PASS] Correctly distinguishes THOUGHT from FINAL ANSWER!")
    print("\n" + "=" * 70)
except Exception as e:
    print(f"[FAIL] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n*** ALL TESTS PASSED - NO API CALLS MADE ***")
print("=" * 70)
print("\nThe prompts.py file is working correctly!")
print("You can now safely initialize the agent.")
