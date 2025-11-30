# SQL ReAct Agent - Build Log

**Purpose:** Document all design decisions, tradeoffs, scraped ideas, and learning insights as we build this agent from scratch.

---

## Project Context

- **Goal:** Build a SQL ReAct agent without using agent frameworks (no LangChain, SmolAgents, etc.)
- **Constraint:** 350-400 lines of Python (excluding comments/docstrings)
- **LLM:** Using Groq or Gemini (free APIs, not OpenAI/Anthropic)
- **Database:** Simple employee-department SQLite database (read-only)
- **Learning Focus:** Understand what agent frameworks do under the hood by building from scratch

---

## Phase 1: Database Setup

### Decision 1: Database Schema
**Choice:** Classic employee-department schema (2 tables)
**Rationale:**
- ✅ Simple to understand and debug
- ✅ Covers all SQL operations needed (COUNT, AVG, GROUP BY, JOIN)
- ✅ Real-world use case (HR/business questions)
- ✅ Easy to create test cases
- ❌ Initially considered e-commerce, but more complex (needs junction tables)

**Tradeoff:** Start simple (2 tables), can expand later if needed for testing complex queries.

---

### Decision 2: Data Storage - CSV vs Hardcoded
**Choice:** CSV files with hardcoded fallback
**Rationale:**
- ✅ Easy to expand dataset (edit CSV, don't touch code)
- ✅ Portable (falls back to hardcoded if CSV missing)
- ✅ Good engineering practice (separate data from logic)
- ✅ Clean structure for sharing/collaboration

**Implementation:**
```python
# data.py tries CSV first, falls back to DEPARTMENTS_FALLBACK
departments = load_departments(script_dir)  # CSV or fallback
```

**Alternative Considered:** Just hardcoded lists
- ❌ Would work, but harder to expand dataset
- ❌ Mixes data with logic

**Verdict:** CSV + fallback wins. Minimal extra code (~30 lines) for better maintainability.

---

### Decision 3: Project Structure
**Choice:**
```
sql_react_agent/
├── database/
│   ├── schema.sql          # Pure SQL schema
│   ├── data.py             # Population script
│   ├── sample_data/        # CSV files
│   └── company.db          # Generated DB
├── src/                    # Agent code
├── tests/                  # Tests
└── README.md
```

**Rationale:**
- ✅ Clear separation of concerns
- ✅ Database is self-contained (can test independently)
- ✅ Easy to navigate
- ✅ Standard Python project layout

---

## Phase 2: Tool System (In Progress)

### Decision 4: Tool Implementation Approach
**Question:** Should we use plain functions or a Tool class?

**Option 1: Plain Functions + Manual Registry**
```python
def list_tables(db_path): ...
def describe_table(db_path, table_name): ...

TOOLS = {"list_tables": list_tables, ...}
TOOL_DESCRIPTIONS = "..." # Manually write for prompt
```

**Pros:**
- ✅ Simplest possible (~0 extra lines)
- ✅ No abstraction overhead
**Cons:**
- ❌ Tool descriptions separate from functions
- ❌ Hard to iterate over tools for prompt generation
- ❌ Manual tracking of parameter schemas

**Option 2: Minimal Tool Class**
```python
class Tool:
    def __init__(self, name, func, description, param_schema):
        self.name = name
        self.func = func  # Function pointer
        self.description = description
        self.param_schema = param_schema
    
    def __call__(self, **kwargs):
        return self.func(**kwargs)

tools = [
    Tool("list_tables", list_tables, "Lists all tables", {}),
    Tool("describe_table", describe_table, "Describes table", {"table_name": "str"}),
]
```

**Pros:**
- ✅ Standardizes tool structure
- ✅ Easy to generate prompts (iterate over tools)
- ✅ Keeps function + metadata together
- ✅ Easy to add new tools
- ✅ Still very simple (~15 lines)

**Cons:**
- ❌ Minimal extra code (~10-15 lines)

**Choice:** **Option 2 (Minimal Tool Class)**
**Rationale:** 
- Better organization for ~15 lines of code
- Makes prompt generation cleaner
- Standard pattern (similar to what LangChain does, but minimal)
- Still well under LOC limit

---

### Understanding: Tool Class vs LangChain's BaseTool

**What LangChain's BaseTool Does:**
```python
class BaseTool:
    name: str
    description: str
    args_schema: Type[BaseModel]  # Pydantic schema
    
    def _run(self, **kwargs):
        # Implementation
    
    # Additional features:
    # - Automatic validation (via Pydantic)
    # - Error handling
    # - Async support
    # - Callbacks/logging
    # - Return type checking
```

**Our Minimal Tool Class:**
```python
class Tool:
    name: str
    func: callable
    description: str
    param_schema: dict  # Just for documentation/prompts
    
    def __call__(self, **kwargs):
        return self.func(**kwargs)
```

**Key Insight:** 
- LangChain's `BaseTool` is just **standardization** + **extra features**
- Core purpose: Keep function + metadata (name, description, params) together
- Our version: Same core idea, but minimal (no validation, no async, etc.)
- **We're building the 20% that gives 80% of the value**

---

## Design Principles We're Following

1. **Simplicity First:** Start with minimal working version
2. **No Premature Abstraction:** Only add complexity when needed
3. **Code Budget Awareness:** Keep under 350-400 LOC
4. **Learning Over Features:** Build to understand, not to ship production code
5. **Iterative Development:** Build → Test → Refine (agile approach)
6. **Separation of Concerns:** Database, Tools, Agent, Prompts kept separate

---

## Scraped Ideas

### ❌ Using LangChain (Even Just for LLMs)
**Why Scraped:** Assignment explicitly prohibits agent frameworks, including LangChain
**Learning Value:** Forces us to use raw OpenAI/Anthropic/Groq/Gemini SDK

### ❌ Complex Database (E-commerce with Junction Tables)
**Why Scraped:** Start simple (employee-department), can expand later if needed
**Learning Value:** Simpler schema = easier debugging during agent development

### ❌ Schema Validation (Pydantic)
**Why Scraped:** Not needed for assignment, adds LOC, we'll do string parsing
**Learning Value:** Understand how to parse/validate manually

---

## Next Steps

- [ ] Build minimal Tool class
- [ ] Implement 3 tool functions (list_tables, describe_table, query_database)
- [ ] Add SQL validation (SELECT-only, identifier whitelisting)
- [ ] Test tools independently before integrating with LLM

---

## Lessons Learned (Updated as We Build)

1. **Tool Class = Standardization:** Same concept as LangChain's BaseTool, just minimal
2. **CSV + Fallback Pattern:** Good engineering even if not needed for assignment
3. **Start Simple, Iterate:** Build MVP first (2 tables), expand if needed

---

## Questions & Answers

**Q:** Do we need decorators for tools?
**A:** No. Decorators are syntactic sugar. We just need function pointers + metadata.

**Q:** Can't we use functions directly without a Tool class?
**A:** Yes! But Tool class makes it easier to:
   - Generate prompts (iterate over tools)
   - Keep function + description together
   - Add new tools consistently

**Q:** Is this the same as LangChain's BaseTool?
**A:** Core concept: Yes (standardization). Implementation: Simpler (no Pydantic, async, etc.)

---

*This document will be updated as we make more decisions and learn throughout the build process.*

---

## Phase 2: Tool System Implementation (COMPLETE)

### Decision 5: Tool Testing Strategy
**Date:** 2025-11-30

**Approach:** Test tools independently before LLM integration

**Why:**
- ✅ Verify each tool works correctly in isolation
- ✅ Catch bugs early (easier to debug without LLM in the mix)
- ✅ Build confidence in foundation before adding complexity
- ✅ Can test SQL validation robustly

**Testing Results (all tests passed):**

**Test 1: list_tables()**
- ✓ Successfully lists all tables
- ✓ Returns: `departments, employees, sqlite_sequence`

**Test 2: describe_table()**
- ✓ Correctly describes employee table schema
- ✓ Shows columns with types and constraints
- ✓ Includes row count
- ✓ Properly handles invalid table names with helpful error

**Test 3: query_database() - Valid Queries**
- ✓ Simple SELECT works
- ✓ GROUP BY with aggregation works
- ✓ JOIN works correctly
- ✓ Results formatted as readable tables

**Test 4: SQL Validation**
- ✓ Blocks DELETE queries
- ✓ Blocks DROP TABLE
- ✓ Blocks INSERT
- ✓ Blocks UPDATE  
- ✓ Blocks semicolon injection (`SELECT...; DROP...`)
- ✓ All dangerous queries properly rejected

**Test 5: Schema Extraction**
- ✓ Extracts full schema for LLM context
- ✓ Includes table names, columns, types, constraints, row counts

**Test 6: Auto-LIMIT**
- ✓ Automatically adds LIMIT 100 to queries without LIMIT
- ✓ Doesn't add if LIMIT already present

**Implementation Complete:**
- `tools.py`: 339 lines
  - Tool class (standardization wrapper)
  - 4-layer SQL validation
  - 3 tool functions (list_tables, describe_table, query_database)
  - Schema extraction utility
  - Tool registry creation

---

## Phase 3: Agent Implementation (NEXT)

**Upcoming Work:**
- [ ] Create `prompts.py` - System prompt + few-shot examples ✅ (DONE)
- [ ] Create `agent.py` - ReAct loop + LLM integration  
- [ ] Test with actual LLM (Groq or Gemini)
- [ ] End-to-end integration test

**Important Documentation:**
- See [`docs/REACT_LOOP_INTERNALS.md`](../docs/REACT_LOOP_INTERNALS.md) for deep dive into:
  - How ReAct loops actually work (interruption-based execution)
  - The hallucination problem and 4 strategies to prevent it
  - Comparison with LangChain's `invoke()` and `bind_tools`
  - Conversation history management
  - Why manual implementation gives better understanding

---
