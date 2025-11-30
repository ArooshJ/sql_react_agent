"""
Flask Backend API for SQL ReAct Agent

Clean backend that handles agent logic and returns FULL ReAct steps.
NO truncation - every message is returned in full.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import create_agent
from prompts import extract_action_from_response, has_final_answer, extract_final_answer
from tools import list_tables, describe_table, query_database

app = Flask(__name__)
CORS(app)  # Allow Streamlit frontend to call this API

# Global agent instance
agent = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "agent_ready": agent is not None})


@app.route('/init', methods=['POST'])
def initialize_agent():
    """Initialize the agent with config."""
    global agent
    
    data = request.json
    db_path = data.get('db_path', 'database/company.db')
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY not found"}), 500
    
    try:
        agent = create_agent(
            db_path=db_path,
            api_key=api_key,
            llm_provider="groq",
            verbose=False,
            min_delay_between_calls=20.0
        )
        
        return jsonify({
            "status": "initialized",
            "db_path": db_path,
            "schema": agent.schema
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Execute a query and return FULL ReAct loop steps.
    NO TRUNCATION - returns complete messages.
    """
    if agent is None:
        return jsonify({"error": "Agent not initialized. Call /init first."}), 400
    
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Run query with FULL logging (no truncation!)
    try:
        result = run_query_with_full_logging(agent, user_query)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "final_answer": f"Error: {str(e)}",
            "steps": []
        }), 500


def run_query_with_full_logging(agent, query):
    """
    Run query and capture ALL ReAct steps with COMPLETE messages.
    NO ELLIPSES (...) - full LLM outputs returned.
    """
    # Create fresh DB connection (threading fix)
    db_conn = sqlite3.connect(agent.db_path)
    
    steps = []
    
    # Initialize conversation
    history = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": query}
    ]
    
    # ReAct loop
    for iteration in range(1, agent.config.max_iterations + 1):
        step_data = {
            "iteration": iteration,
            "type": None,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "thought": "",
            "action": "",
            "action_params": {},
            "observation": "",
            "raw_llm_response": ""  # FULL response, no truncation
        }
        
        try:
            # Call LLM
            llm_response = agent._call_llm(history)
            
            # Store FULL response (NO truncation!)
            step_data["raw_llm_response"] = llm_response
            
            # Add to history
            history.append({"role": "assistant", "content": llm_response})
            
            # Check for final answer
            if has_final_answer(llm_response):
                final_answer = extract_final_answer(llm_response)
                step_data["type"] = "FINAL_ANSWER"
                step_data["final_answer"] = final_answer
                steps.append(step_data)
                
                db_conn.close()
                return {
                    "final_answer": final_answer,
                    "steps": steps,
                    "iterations": iteration,
                    "status": "success"
                }
            
            # Parse action
            action_name, params = extract_action_from_response(llm_response)
            
            if not action_name:
                step_data["type"] = "ERROR"
                step_data["observation"] = "No valid ACTION found in LLM response"
                steps.append(step_data)
                continue
            
            # Extract THOUGHT (everything before ACTION)
            if "THOUGHT:" in llm_response:
                thought = llm_response.split("ACTION:")[0].replace("THOUGHT:", "").strip()
                step_data["thought"] = thought  # FULL thought, no truncation
            
            # Execute tool with fresh connection
            step_data["type"] = "REACT_CYCLE"
            step_data["action"] = action_name
            step_data["action_params"] = params
            
            try:
                if action_name == "list_tables":
                    observation = list_tables(db_conn)
                elif action_name == "describe_table":
                    observation = describe_table(db_conn, **params)
                elif action_name == "query_database":
                    observation = query_database(db_conn, **params)
                else:
                    observation = f"Unknown tool: {action_name}"
            except Exception as e:
                observation = f"Tool error: {str(e)}"
            
            # Store FULL observation (no truncation!)
            step_data["observation"] = observation
            
            steps.append(step_data)
            
            # Add observation to history
            history.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            
        except Exception as e:
            step_data["type"] = "ERROR"
            step_data["observation"] = str(e)
            steps.append(step_data)
            break
    
    db_conn.close()
    
    return {
        "final_answer": "Max iterations reached without final answer",
        "steps": steps,
        "iterations": agent.config.max_iterations,
        "status": "max_iterations"
    }


if __name__ == '__main__':
    print("="*70)
    print("SQL REACT AGENT - FLASK API")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health       - Health check")
    print("  POST /init         - Initialize agent")
    print("  POST /query        - Execute query (returns FULL ReAct steps)")
    print("\nStarting server...")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
