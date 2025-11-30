"""
Tool System for SQL ReAct Agent

This module provides:
1. Tool class - standardizes tool structure
2. SQL validation - ensures safe, read-only queries
3. Database tools - list_tables, describe_table, query_database
4. Schema extraction - for efficient LLM context
"""

import sqlite3
import re
from typing import Callable, Dict, Tuple
from functools import partial


class Tool:
    """
    Standardized tool wrapper for agent functions.
    
    Similar to LangChain's BaseTool but minimal - just stores function + metadata.
    No validation, no async, no Pydantic - just standardization.
    """
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        param_schema: Dict[str, str]
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name (must match what LLM will call)
            func: The actual function to execute
            description: Human-readable description for LLM prompt
            param_schema: Dict of param_name -> type (for LLM prompt)
                         e.g., {"table_name": "str", "query": "str"}
        """
        self.name = name
        self.func = func
        self.description = description
        self.param_schema = param_schema
    
    def __call__(self, **kwargs) -> str:
        """Make tool callable like a function."""
        return self.func(**kwargs)
    
    def to_prompt_string(self) -> str:
        """Generate a prompt-friendly description of this tool."""
        params_str = ", ".join([f"{k} ({v})" for k, v in self.param_schema.items()])
        if params_str:
            return f"{self.name}({params_str}): {self.description}"
        else:
            return f"{self.name}(): {self.description}"


# ==============================================================================
# SQL VALIDATION - Multi-layer validation for safe SQL execution
# ==============================================================================

def validate_sql_query(query: str, db_conn) -> Tuple[bool, str]:
    """
    Validates SQL query is safe and read-only using multi-layer approach.
    
    Validation layers:
    1. Must start with SELECT or WITH (CTEs allowed)
    2. Single statement only (no semicolons in middle)
    3. Keyword blacklist (dangerous operations)
    4. SQLite EXPLAIN validation (syntax check without execution)
    
    Args:
        query: SQL query to validate
        db_conn: Active SQLite database connection
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    query_stripped = query.strip()
    query_upper = query_stripped.upper()
    
    # Layer 1: Must start with SELECT or WITH (for CTEs)
    valid_starts = ['SELECT', 'WITH']
    if not any(query_upper.startswith(start) for start in valid_starts):
        return False, "Only SELECT queries (including WITH/CTEs) are allowed."
    
    # Layer 2: No multiple statements (semicolon in middle)
    query_no_trailing = query_stripped.rstrip(';').rstrip()
    if ';' in query_no_trailing:
        return False, "Multiple statements not allowed."
    
    # Layer 3: Keyword blacklist (defense in depth)
    dangerous_keywords = [
        'DELETE', 'DROP', 'INSERT', 'UPDATE', 'ALTER', 
        'TRUNCATE', 'EXEC', 'EXECUTE', 'CREATE', 'REPLACE', 
        'PRAGMA', 'ATTACH', 'DETACH'
    ]
    
    for keyword in dangerous_keywords:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, query_upper):
            return False, f"Dangerous keyword '{keyword}' detected."
    
    # Layer 4: Use SQLite's EXPLAIN to validate syntax
    try:
        cursor = db_conn.cursor()
        cursor.execute(f"EXPLAIN {query}")
        return True, ""
    except sqlite3.Error as e:
        return False, f"Invalid SQL syntax: {str(e)}"


def add_limit_if_missing(query: str, default_limit: int = 100) -> str:
    """Adds LIMIT clause to query if not present."""
    if 'LIMIT' in query.upper():
        return query
    query_stripped = query.rstrip(';').rstrip()
    return f"{query_stripped} LIMIT {default_limit}"


def sanitize_identifier(identifier: str) -> str:
    """Sanitizes table/column names to prevent SQL injection."""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(
            f"Invalid identifier '{identifier}'. "
            "Only alphanumeric characters and underscores allowed."
        )
    return identifier


# ==============================================================================
# TOOL FUNCTIONS - The 3 required tools for SQL database interaction
# ==============================================================================

def list_tables(db_conn) -> str:
    """Lists all tables in the database."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        if not tables:
            return "No tables found in database."
        
        table_names = [table[0] for table in tables]
        return f"Tables in database: {', '.join(table_names)}"
        
    except sqlite3.Error as e:
        return f"Error listing tables: {str(e)}"


def describe_table(db_conn, table_name: str) -> str:
    """Describes the schema of a specific table."""
    try:
        # Sanitize table name
        try:
            safe_table_name = sanitize_identifier(table_name)
        except ValueError as e:
            return f"Error: {str(e)}"
        
        cursor = db_conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (safe_table_name,))
        
        if not cursor.fetchone():
            return f"Error: Table '{table_name}' does not exist."
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({safe_table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {safe_table_name}")
        row_count = cursor.fetchone()[0]
        
        # Format schema information
        schema_lines = [f"Table: {table_name}"]
        schema_lines.append(f"Row count: {row_count}")
        schema_lines.append("Columns:")
        
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            not_null = " NOT NULL" if col[3] else ""
            pk = " PRIMARY KEY" if col[5] else ""
            schema_lines.append(f"  - {col_name} ({col_type}){not_null}{pk}")
        
        return "\n".join(schema_lines)
        
    except sqlite3.Error as e:
        return f"Error describing table: {str(e)}"


def query_database(db_conn, query: str) -> str:
    """Executes a read-only SQL SELECT query on the database."""
    try:
        # Validate query is safe
        is_valid, error_msg = validate_sql_query(query, db_conn)
        if not is_valid:
            return f"Query validation failed: {error_msg}"
        
        # Add LIMIT if missing
        query_with_limit = add_limit_if_missing(query, default_limit=100)
        
        # Execute query
        cursor = db_conn.cursor()
        cursor.execute(query_with_limit)
        results = cursor.fetchall()
        
        # Format results
        if not results:
            return "Query executed successfully but returned no results."
        
        column_names = [desc[0] for desc in cursor.description]
        
        result_lines = []
        result_lines.append(f"Query returned {len(results)} row(s):")
        result_lines.append("")
        
        # Header
        header = " | ".join(column_names)
        result_lines.append(header)
        result_lines.append("-" * len(header))
        
        # Rows (limit display to first 20 for readability)
        display_limit = min(20, len(results))
        for row in results[:display_limit]:
            row_str = " | ".join(str(val) for val in row)
            result_lines.append(row_str)
        
        if len(results) > display_limit:
            result_lines.append(f"... ({len(results) - display_limit} more rows not shown)")
        
        return "\n".join(result_lines)
        
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ==============================================================================
# SCHEMA EXTRACTION - For efficient context (extract once, reuse)
# ==============================================================================

def extract_full_schema(db_conn) -> str:
    """
    Extracts complete database schema for LLM context.
    
    This is run once at initialization and included in system prompt.
    """
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        schema_parts = ["DATABASE SCHEMA:"]
        schema_parts.append("")
        
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            schema_parts.append(f"Table: {table_name} ({row_count} rows)")
            
            col_info = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                constraints = []
                if col[5]:  # Primary key
                    constraints.append("PK")
                if col[3]:  # Not null
                    constraints.append("NOT NULL")
                
                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
                col_info.append(f"{col_name} ({col_type}){constraint_str}")
            
            schema_parts.append("  Columns: " + ", ".join(col_info))
            schema_parts.append("")
        
        return "\n".join(schema_parts)
        
    except sqlite3.Error as e:
        return f"Error extracting schema: {str(e)}"


# ==============================================================================
# TOOL REGISTRY - Create standardized Tool instances
# ==============================================================================

def create_tool_registry(db_conn):
    """
    Creates the tool registry for the agent.
    
    Returns both a list (for iteration) and dict (for lookup by name).
    """
    tools = [
        Tool(
            name="list_tables",
            func=partial(list_tables, db_conn),
            description="Lists all tables in the database. No parameters required.",
            param_schema={}
        ),
        Tool(
            name="describe_table",
            func=partial(describe_table, db_conn),
            description="Describes the schema of a specific table including columns, types, and row count.",
            param_schema={"table_name": "str"}
        ),
        Tool(
            name="query_database",
            func=partial(query_database, db_conn),
            description="Executes a read-only SELECT query on the database. Only SELECT queries allowed.",
            param_schema={"query": "str"}
        ),
    ]
    
    tools_dict = {tool.name: tool for tool in tools}
    
    return tools, tools_dict