"""
Example usage of the SQL ReAct Agent

This script shows how to use the agent to query a database
using natural language.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import create_agent

def main():
    """Run example queries against the agent."""
    
    # Path to database
    db_path = os.path.join(os.path.dirname(__file__), 'database', 'company.db')
    
    # Get API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment")
        print("Set it with: export GROQ_API_KEY='your-key-here'")
        return
    
    # Create agent with context manager (auto-closes DB connection)
    with create_agent(
        db_path=db_path,
        api_key=api_key,
        llm_provider="groq",
        verbose=True  # Show detailed output
    ) as agent:
        
        # Example queries
        queries = [
            "How many employees are in the database?",
            "What's the average salary by department?",
            "How many people work in Engineering?",
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n\n{'#'*70}")
            print(f"# QUERY {i}/{len(queries)}")
            print(f"{'#'*70}\n")
            
            result = agent.run(query)
            
            print(f"\n{'='*70}")
            print("FINAL RESULT:")
            print(f"{'='*70}")
            print(result)
            
            # Wait for user between queries
            if i < len(queries):
                input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    main()
