"""
Manual test script for tools.py

Tests all 3 tools and validation functions independently before integrating with LLM.
"""

import sqlite3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import (
    create_tool_registry,
    extract_full_schema,
    validate_sql_query
)

def test_tools():
    """Test all tools with our sample database."""
    
    # Connect to database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'company.db')
    db_conn = sqlite3.connect(db_path)
    print(f"Connected to: {db_path}\n")
    
    # Create tools
    tools, tools_dict = create_tool_registry(db_conn)
    print("✓ Tools created successfully")
    print(f"  Available tools: {list(tools_dict.keys())}\n")
    
    # ==================================================================
    # Test 1: list_tables
    # ==================================================================
    print("="*60)
    print("TEST 1: list_tables()")
    print("="*60)
    result = tools_dict['list_tables']()
    print(result)
    print()
    
    # ==================================================================
    # Test 2: describe_table
    # ==================================================================
    print("="*60)
    print("TEST 2: describe_table(table_name='employees')")
    print("="*60)
    result = tools_dict['describe_table'](table_name='employees')
    print(result)
    print()
    
    print("="*60)
    print("TEST 2b: describe_table with invalid table")
    print("="*60)
    result = tools_dict['describe_table'](table_name='nonexistent')
    print(result)
    print()
    
    # ==================================================================
    # Test 3: query_database - Valid queries
    # ==================================================================
    print("="*60)
    print("TEST 3: query_database - Simple SELECT")
    print("="*60)
    query = "SELECT * FROM employees"
    print(f"Query: {query}")
    result = tools_dict['query_database'](query=query)
    print(result)
    print()
    
    print("="*60)
    print("TEST 3b: query_database - AVG with GROUP BY")
    print("="*60)
    query = "SELECT department_id, AVG(salary) as avg_salary FROM employees GROUP BY department_id"
    print(f"Query: {query}")
    result = tools_dict['query_database'](query=query)
    print(result)
    print()
    
    print("="*60)
    print("TEST 3c: query_database - JOIN")
    print("="*60)
    query = """
    SELECT e.name, d.name as dept_name, e.salary 
    FROM employees e 
    JOIN departments d ON e.department_id = d.id
    """
    print(f"Query: {query.strip()}")
    result = tools_dict['query_database'](query=query)
    print(result)
    print()
    
    # ==================================================================
    # Test 4: SQL Validation - Should REJECT dangerous queries
    # ==================================================================
    print("="*60)
    print("TEST 4: SQL Validation - Dangerous queries (should fail)")
    print("="*60)
    
    dangerous_queries = [
        "DELETE FROM employees WHERE id = 1",
        "DROP TABLE employees",
        "INSERT INTO employees VALUES (99, 'Hacker', 'hack@evil.com', 1, 99999, '2025-01-01')",
        "UPDATE employees SET salary = 0",
        "SELECT * FROM employees; DROP TABLE employees;",
    ]
    
    for i, query in enumerate(dangerous_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = tools_dict['query_database'](query=query)
        print(f"   Result: {result}")
    
    print()
    
    # ==================================================================
    # Test 5: Schema Extraction
    # ==================================================================
    print("="*60)
    print("TEST 5: extract_full_schema()")
    print("="*60)
    schema = extract_full_schema(db_conn)
    print(schema)
    print()
    
    # ==================================================================
    # Test 6: LIMIT auto-add
    # ==================================================================
    print("="*60)
    print("TEST 6: Auto-add LIMIT (query without LIMIT)")
    print("="*60)
    query = "SELECT * FROM employees"
    print(f"Query: {query}")
    print("(Should automatically add LIMIT 100)")
    result = tools_dict['query_database'](query=query)
    # Result should indicate it was limited
    print(f"First line of result: {result.splitlines()[0]}")
    print()
    
    # Close connection
    db_conn.close()
    print("="*60)
    print("✓ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_tools()
