"""
Simple single-query test for SQL ReAct Agent

This script tests the agent with ONE query to verify it works
and to test rate limiting before running multiple queries.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from parent directory
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Set GROQ_API_KEY manually.")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import create_agent

def main():
    """Run a single test query."""
    
    # Path to database
    db_path = os.path.join(os.path.dirname(__file__), 'database', 'company.db')
    
    # Get API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment")
        print("Make sure .env file is loaded or set: export GROQ_API_KEY='your-key-here'")
        return
    
    
    print("\n" + "="*70)
    print("SINGLE QUERY TEST - SQL ReAct Agent")
    print("="*70)
    print("\nPROTECTIONS ENABLED:")
    print("  - Rate limiting: 2s between API calls")
    print("  - Daily token limit detection (stops immediately if hit)")
    print("  - Exponential backoff on per-minute rate limits")
    print("\nThis will run ONE query safely.")
    print("If successful, you can run example_usage.py for more tests.\n")
    
    # Create agent with context manager (auto-closes DB connection)
    with create_agent(
        db_path=db_path,
        api_key=api_key,
        llm_provider="groq",
        verbose=True,  # Show detailed output
        min_delay_between_calls=2.0  # 2-second safety delay
    ) as agent:
        
        # Single simple query
        query = "How many employees are in the database?"
        
        print(f"\nQuery: {query}\n")
        result = agent.run(query)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT:")
        print(f"{'='*70}")
        print(result)
        print(f"{'='*70}\n")
    
    print("✓ Test complete! If successful, you can now run example_usage.py")


if __name__ == "__main__":
    main()
