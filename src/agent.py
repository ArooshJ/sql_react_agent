"""
SQL ReAct Agent Implementation

This module contains the SQLReActAgent class that implements the ReAct
(Reasoning + Acting) loop for querying databases using natural language.

Key features:
- Multi-turn conversation with LLM
- Tool execution and observation injection  
- Stop sequences to prevent hallucinations
- Rate limiting to prevent API suspension
- Comprehensive error handling
- Conversation history management
"""

import sqlite3
import os
import time
from typing import Optional, Dict, List
from dataclasses import dataclass

# Import our modules
from tools import create_tool_registry, extract_full_schema
from prompts import (
    build_system_prompt,
    extract_action_from_response,
    has_final_answer,
    extract_final_answer
)


@dataclass
class AgentConfig:
    """Configuration for SQL ReAct Agent."""
    max_iterations: int = 10
    temperature: float = 0.0  # Deterministic for reliability
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq model
    verbose: bool = True
    stop_sequences: List[str] = None
    # Rate limiting to prevent API suspension - CRITICAL!
    min_delay_between_calls: float = 20.0  # 5 seconds between calls (reasonable for testing)
    max_retries: int = 3  # Maximum retries on error
    retry_delay: float = 5.0  # Initial retry delay (exponential backoff)
    
    def __post_init__(self):
        if self.stop_sequences is None:
            # Default stop sequences to prevent hallucinations
            self.stop_sequences = ["OBSERVATION:", "\nOBSERVATION", "Observation:"]


class SQLReActAgent:
    """
    SQL Database Agent using ReAct (Reasoning + Acting) pattern.
    
    This agent can answer natural language questions about a SQL database
    by iteratively reasoning about what to do and executing SQL queries.
    
    Example:
        >>> agent = SQLReActAgent("database/company.db", api_key="...")
        >>> result = agent.run("How many employees work in Engineering?")
        >>> print(result)
    """
    
    def __init__(
        self,
        db_path: str,
        api_key: str,
        config: Optional[AgentConfig] = None,
        llm_provider: str = "groq"  # "groq" or "gemini"
    ):
        """
        Initialize the SQL ReAct Agent.
        
        Args:
            db_path: Path to SQLite database file
            api_key: API key for LLM provider
            config: Agent configuration (uses defaults if None)
            llm_provider: LLM provider to use ("groq" or "gemini")
        """
        self.db_path = db_path
        self.config = config or AgentConfig()
        self.llm_provider = llm_provider
        
        # Initialize database connection
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.db_conn = sqlite3.connect(db_path)
        
        # Create tools
        self.tools, self.tools_dict = create_tool_registry(self.db_conn)
        
        # Extract schema for context
        self.schema = extract_full_schema(self.db_conn)
       
        # Build system prompt
        self.system_prompt = build_system_prompt(
            schema=self.schema,
            tools=self.tools,
            include_examples=True
        )
        
        # Initialize LLM client
        self._init_llm(api_key)
        
        # Track last API call time for rate limiting
        self._last_api_call = 0
        
        if self.config.verbose:
            print("✓ SQL ReAct Agent initialized")
            print(f"  Database: {db_path}")
            print(f"  LLM Provider: {llm_provider}")
            print(f"  Model: {self.config.model_name}")
            print(f"  Tools: {list(self.tools_dict.keys())}")
            print(f"  ⚠️  Rate Limit: {self.config.min_delay_between_calls}s between API calls")
    
    def _init_llm(self, api_key: str):
        """Initialize LLM client based on provider."""
        if self.llm_provider == "groq":
            from groq import Groq
            self.llm = Groq(api_key=api_key)
        elif self.llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(self.config.model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call LLM with messages and return response.
        
        Implements multiple safeguards:
        1. Stop sequences to prevent hallucinations
        2. Rate limiting to prevent API suspension
        3. Retry logic with exponential backoff
        4. Daily token limit detection (stops immediately if hit)
        
        Args:
            messages: Conversation history
            
        Returns:
            LLM response text
            
        Raises:
            RuntimeError: If daily token limit is hit or max retries exceeded
        """
        # Rate limiting: Ensure minimum delay between API calls
        time_since_last_call = time.time() - self._last_api_call
        if time_since_last_call < self.config.min_delay_between_calls:
            sleep_time = self.config.min_delay_between_calls - time_since_last_call
            if self.config.verbose:
                print(f"⏳ Rate limit: Waiting {sleep_time:.1f}s before next API call...")
            time.sleep(sleep_time)
        
        # Retry logic with exponential backoff
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if self.config.verbose and attempt > 1:
                    print(f"🔄 Retry attempt {attempt}/{self.config.max_retries}")
                
                # Record call time
                self._last_api_call = time.time()
                
                # Make API call
                if self.llm_provider == "groq":
                    response = self.llm.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        stop=self.config.stop_sequences  # ← Stop before OBSERVATION!
                    )
                    return response.choices[0].message.content
                
                elif self.llm_provider == "gemini":
                    # Convert messages to Gemini format
                    prompt_text = "\n".join([
                        f"{msg['role']}: {msg['content']}" for msg in messages[1:]
                    ])
                    response = self.llm.generate_content(prompt_text)
                    return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for DAILY token limit (DO NOT RETRY!)
                if any(keyword in error_str for keyword in [
                    'daily limit',
                    'quota exceeded',
                    'daily quota',
                    'rate_limit_exceeded'  # Groq specific
                ]):
                    error_msg = (
                        "\n" + "="*70 + "\n"
                        "🛑 DAILY TOKEN LIMIT REACHED!\n"
                        "="*70 + "\n"
                        "Groq daily token limit has been exceeded.\n"
                        "This limit resets daily (usually at midnight UTC).\n\n"
                        "Actions:\n"
                        "1. Wait for limit reset (check Groq dashboard)\n"
                        "2. Or switch to another provider (set llm_provider='gemini')\n"
                        "3. Or use a different API key if available\n\n"
                        f"Original error: {str(e)}\n"
                        "="*70
                    )
                    if self.config.verbose:
                        print(error_msg)
                    raise RuntimeError(error_msg)
                
                # Check for per-minute rate limit (CAN RETRY!)
                if any(keyword in error_str for keyword in [
                    'rate limit',
                    'too many requests',
                    'requests per minute'
                ]):
                    if self.config.verbose:
                        print(f"⚠️  Rate limit hit (requests/minute)")
                    
                    if attempt < self.config.max_retries:
                        retry_delay = self.config.retry_delay * (2 ** (attempt - 1))
                        if self.config.verbose:
                            print(f"⏳ Backing off: Waiting {retry_delay:.1f}s before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise RuntimeError(f"Rate limit persists after {self.config.max_retries} retries")
                
                # Other errors - retry with backoff
                if attempt < self.config.max_retries:
                    retry_delay = self.config.retry_delay * (2 ** (attempt - 1))
                    if self.config.verbose:
                        print(f"❌ API Error: {str(e)}")
                        print(f"⏳ Waiting {retry_delay:.1f}s before retry...")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed
                    raise
        
        raise RuntimeError("Failed to call LLM after max retries")
    
    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.
        
        This implements the ReAct loop:
        1. Call LLM (gets THOUGHT + ACTION or FINAL ANSWER)
        2. If ACTION: execute tool, add OBSERVATION, repeat
        3. If FINAL ANSWER: return to user
        
        Args:
            user_query: Natural language question about the database
            
        Returns:
            Final answer string
        """
        # Initialize conversation history
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        if self.config.verbose:
            print("\n" + "="*70)
            print(f"USER QUERY: {user_query}")
            print("="*70)
        
        # Main ReAct loop
        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                print(f"\n{'─'*70}")
                print(f"ITERATION {iteration}")
                print(f"{'─'*70}")
            
            # ================================================================
            # STEP 1: Call LLM to plan next action
            # ================================================================
            try:
                response = self._call_llm(history)
            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                if self.config.verbose:
                    print(f"❌ {error_msg}")
                return f"Error: {error_msg}"
            
            if self.config.verbose:
                print(f"\nLLM Response:")
                print(response)
            
            # Add assistant message to history
            history.append({
                "role": "assistant",
                "content": response
            })
            
            # ================================================================
            # STEP 2: Check if done (has FINAL ANSWER)
            # ================================================================
            if has_final_answer(response):
                final_answer = extract_final_answer(response)
                
                if self.config.verbose:
                    print("\n" + "="*70)
                    print("✓ FINAL ANSWER RECEIVED")
                    print("="*70)
                    print(final_answer)
                    print("="*70)
                
                return final_answer
            
            # ================================================================
            # STEP 3: Parse action from response
            # ================================================================
            action_name, params = extract_action_from_response(response)
            
            if not action_name:
                error_msg = "No valid ACTION found in response"
                if self.config.verbose:
                    print(f"\n⚠️  {error_msg}")
                    print("Response should contain either:")
                    print("  - THOUGHT + ACTION")
                    print("  - FINAL ANSWER")
                
                # Give LLM a hint to fix itself
                history.append({
                    "role": "user",
                    "content": "Error: Expected ACTION or FINAL ANSWER. Please provide one."
                })
                continue
            
            if self.config.verbose:
                print(f"\n📋 Parsed Action:")
                print(f"  Tool: {action_name}")
                print(f"  Parameters: {params}")
            
            # ================================================================
            # STEP 4: Execute tool
            # ================================================================
            if action_name not in self.tools_dict:
                error_msg = f"Unknown tool: {action_name}"
                if self.config.verbose:
                    print(f"\n❌ {error_msg}")
                    print(f"Available tools: {list(self.tools_dict.keys())}")
                
                # Provide error as observation
                observation = f"Error: {error_msg}. Available tools: {', '.join(self.tools_dict.keys())}"
            else:
                try:
                    if self.config.verbose:
                        print(f"\n🔧 Executing tool: {action_name}...")
                    
                    tool_result = self.tools_dict[action_name](**params)
                    observation = tool_result
                    
                    if self.config.verbose:
                        print("✓ Tool execution complete")
                        print(f"\nTool Result:")
                        print(tool_result)
                
                except Exception as e:
                    observation = f"Tool execution error: {str(e)}"
                    if self.config.verbose:
                        print(f"\n❌ {observation}")
            
            # ================================================================
            # STEP 5: Inject OBSERVATION into history
            # ================================================================
            # Key: We act as "user" providing the observation
            history.append({
                "role": "user",
                "content": f"OBSERVATION: {observation}"
            })
            
            if self.config.verbose:
                print(f"\n💬 Added OBSERVATION to history")
            
            # Loop continues → LLM will be called again with observation
        
        # ================================================================
        # Max iterations reached
        # ================================================================
        if self.config.verbose:
            print("\n" + "="*70)
            print(f"⚠️  MAX ITERATIONS REACHED ({self.config.max_iterations})")
            print("="*70)
        
        return f"Agent stopped: Reached maximum iterations ({self.config.max_iterations}). Last response:\n\n{response}"
    
    def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
            if self.config.verbose:
                print("\n✓ Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ==============================================================================
# Convenience function for quick usage
# ==============================================================================

def create_agent(
    db_path: str,
    api_key: Optional[str] = None,
    llm_provider: str = "groq",
    **config_kwargs
) -> SQLReActAgent:
    """
    Convenience function to create an agent.
    
    Args:
        db_path: Path to SQLite database
        api_key: LLM API key (reads from env if None)
        llm_provider: "groq" or "gemini"
        **config_kwargs: Additional config parameters
        
    Returns:
        Initialized SQLReActAgent
    """
    # Get API key from env if not provided
    if api_key is None:
        if llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
        elif llm_provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not provided and ${llm_provider.upper()}_API_KEY not found in environment")
    
    # Create config
    config = AgentConfig(**config_kwargs)
    
    return SQLReActAgent(db_path, api_key, config, llm_provider)
