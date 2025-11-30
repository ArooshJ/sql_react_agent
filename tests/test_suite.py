"""
Comprehensive Test Suite for SQL ReAct Agent

Tests queries of varying complexity and logs results.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed.")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import create_agent


# Test queries organized by complexity
TEST_QUERIES = {
    "Simple": [
        "How many employees are in the database?",
        "How many departments do we have?",
    ],
    "Medium": [
        "What's the average salary by department?",
        "List all employees in the Engineering department.",
    ],
    "Complex": [
        "Who is the highest paid employee and which department are they in?",
        "Compare the total salary expense for each department.",
    ]
}


def log_test_result(log_file, complexity, query, result, iterations, success):
    """Log test result to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "[PASS]" if success else "[FAIL]"
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"[{timestamp}] {status}\n")
    log_file.write(f"Complexity: {complexity}\n")
    log_file.write(f"Query: {query}\n")
    log_file.write(f"Iterations: {iterations}\n")
    log_file.write(f"\nResult:\n{result}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()


def run_test_suite():
    """Run comprehensive test suite."""
    
    # Setup
    db_path = os.path.join(os.path.dirname(__file__), 'database', 'company.db')
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return
    
    # Create log file
    log_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(os.path.dirname(__file__), 'logs', log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print("\n" + "="*80)
    print("SQL REACT AGENT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"\nLog file: {log_path}")
    print("\nProtections enabled:")
    print("  - Rate limiting: 2s between API calls")
    print("  - Daily token limit detection")
    print("  - Exponential backoff")
    print("\n" + "="*80)
    
    total_tests = sum(len(queries) for queries in TEST_QUERIES.values())
    current_test = 0
    passed = 0
    failed = 0
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        # Write header
        log_file.write("SQL REACT AGENT - TEST RESULTS\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total Tests: {total_tests}\n")
        log_file.write("="*80 + "\n")
        
        # Create agent (reuse for all tests)
        with create_agent(
            db_path=db_path,
            api_key=api_key,
            llm_provider="groq",
            verbose=False,  # Less verbose for batch testing
            min_delay_between_calls=2.0
        ) as agent:
            
            # Run tests by complexity
            for complexity, queries in TEST_QUERIES.items():
                print(f"\n{'-'*80}")
                print(f"COMPLEXITY: {complexity}")
                print(f"{'-'*80}")
                
                for query in queries:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Query: {query}")
                    print("-" * 80)
                    
                    try:
                        # Track iterations (simple heuristic: count LLM in verbose output)
                        result = agent.run(query)
                        iterations = "Unknown"  # We're not in verbose mode
                        success = "error" not in result.lower()
                        
                        if success:
                            passed += 1
                            status = "[PASS]"
                        else:
                            failed += 1
                            status = "[FAIL]"
                        
                        # Show FULL result
                        print(f"\nStatus: {status}")
                        print(f"\nFull Answer:")
                        print(result)
                        print("\n" + "=" * 80)
                        
                        # Log result
                        log_test_result(log_file, complexity, query, result, iterations, success)
                        
                    except KeyboardInterrupt:
                        print("\n\nTest suite interrupted by user")
                        break
                    except Exception as e:
                        failed += 1
                        print(f"\nStatus: [ERROR]")
                        print(f"Error: {str(e)}")
                        print("\n" + "=" * 80)
                        log_test_result(log_file, complexity, query, f"ERROR: {str(e)}", "N/A", False)
                
                # Small delay between complexity levels
                if complexity != list(TEST_QUERIES.keys())[-1]:
                    print("\n  Pausing 3s before next complexity level...")
                    import time
                    time.sleep(3)
        
        # Write summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total: {total_tests}")
        print(f"Passed: {passed} ({passed/total_tests*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"\nLog saved to: {log_path}")
        print("="*80)
        
        # Write summary to log
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write("SUMMARY\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Total Tests: {total_tests}\n")
        log_file.write(f"Passed: {passed} ({passed/total_tests*100:.1f}%)\n")
        log_file.write(f"Failed: {failed} ({failed/total_tests*100:.1f}%)\n")
        log_file.write(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        run_test_suite()
    except KeyboardInterrupt:
        print("\n\nSuite interrupted. Partial results saved.")
