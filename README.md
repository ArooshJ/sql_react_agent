# SQL ReAct Agent

A SQL Database Agent built using the **ReAct (Reasoning + Acting)** pattern with a clean Flask backend + Streamlit frontend architecture. Query your database using natural language powered by LLMs.

---

## 🏗️ Architecture

**Clean Separation Design:**
- **Backend (Flask API)**: Handles agent logic, LLM calls, and database operations
- **Frontend (Streamlit)**: Interactive UI for queries and results visualization
- **No Threading Issues**: Proper SQLite connection management per request
- **Full Transparency**: Complete ReAct loop (THOUGHT → ACTION → OBSERVATION) displayed

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors requests streamlit groq python-dotenv
```

### 2. Set Up Environment

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your-groq-api-key-here
```

### 3. Start Backend (Terminal 1)

```bash
python api.py
```

Backend runs on `http://localhost:5000`

### 4. Start Frontend (Terminal 2)

```bash
python -m streamlit run app_frontend.py
```

Frontend opens at `http://localhost:8501`

### 5. Use the Application

1. **Auto-Initialization**: The agent initializes automatically when you open the app.
2. **Enter Query**: Type your question in natural language (e.g., "How many employees?").
3. **Run**: Click "Run Query" to execute.
4. **View Reasoning**: Toggle "Show Reasoning" to see the full ReAct loop (Thought → Action → Observation).

---

## 📊 Database Schema

The agent connects to a company database with the following structure:

**departments**
- `id` INTEGER PRIMARY KEY
- `name` TEXT UNIQUE
- `location` TEXT

**employees**
- `id` INTEGER PRIMARY KEY
- `name` TEXT
- `email` TEXT UNIQUE
- `department_id` INTEGER (→ departments.id)
- `salary` REAL
- `hire_date` TEXT (YYYY-MM-DD)

**Sample Data:** 3 departments, 8 employees

---

## 🗂️ Project Structure

```
sql_react_agent/
├── api.py                   # Flask backend API
├── app_frontend.py          # Streamlit frontend UI
├── database/
│   ├── schema.sql           # Database schema definition
│   ├── data.py              # Sample data population
│   └── company.db           # SQLite database file
├── src/
│   ├── tools.py             # Database tools + SQL validation
│   ├── prompts.py           # System prompts + few-shot examples
│   └── agent.py             # ReAct agent core implementation
├── tests/
│   ├── test_tools_manual.py # Database tool tests
│   └── test_prompts.py      # Prompt and parser tests
├── docs/
│   ├── REACT_LOOP_INTERNALS.md
│   ├── LANGCHAIN_INTERNALS.md
│   └── API_SAFETY.md
└── logs/                    # Test execution logs
```

---

## 🎯 Features

### ReAct Pattern Implementation
- Manual ReAct loop without using agent frameworks
- LLM integration with Groq and Google Gemini support
- Conversation history management
- Tool execution with injected observations
- Hallucination prevention via stop sequences

### Interactive UI (Streamlit)
- **Auto-Initialization**: Seamless startup experience
- **Reasoning Toggle**: Option to view full internal reasoning steps or just the final answer
- **Instant Execution**: Optimized request handling for fast responses
- **History Tracking**: Session-based query history

### Database Tools
- **list_tables()** - List all available tables
- **describe_table(table_name)** - Get table schema
- **query_database(query)** - Execute SQL SELECT queries
- **extract_full_schema()** - Get complete database schema

### SQL Safety (4-Layer Validation)
1. **Keyword checking** - Only SELECT statements allowed
2. **Blacklist filtering** - Prevents DROP, DELETE, UPDATE, etc.
3. **Syntax validation** - SQLite EXPLAIN check before execution
4. **Auto-LIMIT** - Automatic LIMIT 100 to prevent data overload
5. **SQL injection prevention** - Parameterized queries

### Intelligent Prompting
- Role-based system prompt
- Auto-generated tool descriptions
- Few-shot examples demonstrating proper ReAct format
- Clear output format instructions
- Turn-by-turn interaction patterns

### API Safety
- Rate limiting (20 seconds between queries)
- Request timeouts with error recovery
- Exponential backoff on failures
- Connection error detection and handling

---

## 📝 Example Queries

**Simple:**
```
"How many employees are in the database?"
"How many departments do we have?"
```

**Medium Complexity:**
```
"What's the average salary by department?"
"List all employees in the Engineering department."
```

**Complex:**
```
"Who is the highest paid employee and which department are they in?"
"Which department has the highest average salary?"
"Compare total salary expenses across all departments."
```

---

## 🔧 API Documentation

### Endpoints

**GET** `/health`
```json
Response: {"status": "healthy"}
```

**POST** `/init`
```json
Request:  {"db_path": "database/company.db"}
Response: {"status": "initialized", "db_path": "...", "schema": "..."}
```

**POST** `/query`
```json
Request:  {"query": "How many employees?"}
Response: {
  "final_answer": "There are 8 employees in the database.",
  "steps": [
    {
      "iteration": 1,
      "type": "REACT_CYCLE",
      "thought": "I need to count the rows in employees table...",
      "action": "query_database",
      "action_params": {"query": "SELECT COUNT(*) FROM employees"},
      "observation": "Query returned 1 row(s): count = 8",
      "raw_llm_response": "THOUGHT: I need to count...\nACTION: ..."
    }
  ],
  "iterations": 1,
  "status": "success"
}
```

---

## 🧪 Testing

### Run Tests

```bash
# Test prompt generation and parsers (no API calls)
python tests/test_prompts.py

# Test database tools manually
python tests/test_tools_manual.py
```

### What's Tested
- ✅ Prompt template building
- ✅ ACTION parsing with single-brace JSON
- ✅ Complex SQL query parsing
- ✅ FINAL ANSWER detection
- ✅ THOUGHT vs FINAL ANSWER distinction
- ✅ Database tool functionality

---

## 🔒 Security & Safety

### Rate Limiting
- 20-second delay between queries
- Configurable rate limits
- Prevents API quota exhaustion

### SQL Injection Prevention
- Whitelist-based query validation
- Syntax pre-validation
- Read-only enforcement
- Auto-limiting result sets

### Error Handling
- Connection timeout management
- Graceful API error recovery
- Detailed error logging

---

## 📖 Documentation

**Detailed Guides:**
- [`docs/REACT_LOOP_INTERNALS.md`](docs/REACT_LOOP_INTERNALS.md) - How the ReAct pattern works
- [`docs/LANGCHAIN_INTERNALS.md`](docs/LANGCHAIN_INTERNALS.md) - LangChain framework comparison
- [`docs/API_SAFETY.md`](docs/API_SAFETY.md) - Rate limiting strategies

---

## 🛠️ Technology Stack

- **Backend:** Flask, SQLite
- **Frontend:** Streamlit
- **LLM Providers:** Groq, Google Gemini
- **Language:** Python 3.8+

---

## 🎓 How It Works

### ReAct Loop Explained

1. **User asks a question** in natural language
2. **LLM generates THOUGHT** (reasoning) and **ACTION** (tool call)
3. **System executes the tool** (e.g., SQL query)
4. **System injects OBSERVATION** back to LLM
5. **LLM continues reasoning** or provides **FINAL ANSWER**
6. **Repeat until answer is found**

### Example Execution Flow

```
User: "How many employees are in Engineering?"

Agent (LLM):
  THOUGHT: Need to find department_id for Engineering first
  ACTION: query_database{"query": "SELECT id FROM departments WHERE name='Engineering'"}

System:
  OBSERVATION: Query returned 1 row: id=1

Agent (LLM):
  THOUGHT: Now count employees with department_id=1
  ACTION: query_database{"query": "SELECT COUNT(*) FROM employees WHERE department_id=1"}

System:
  OBSERVATION: Query returned 1 row: count=3

Agent (LLM):
  FINAL ANSWER: There are 3 employees in the Engineering department.
```

---

## 🔄 Development

### Making Changes

**Backend** (`api.py`, `src/*.py`):
- Flask auto-reloads in debug mode
- Changes take effect immediately

**Frontend** (`app_frontend.py`):
- Streamlit auto-reloads on file save
- Refresh browser to see changes

**Database** (`database/`):
- Modify `schema.sql` or `data.py` and regenerate

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional database tools
- Advanced SQL query support
- Multiple database backend support
- Enhanced UI features
- Additional LLM provider integrations

---

## 🙏 Acknowledgments

Built as part of the Vizuara assignment.

**Tech Stack:** Flask • Streamlit • Groq • SQLite
