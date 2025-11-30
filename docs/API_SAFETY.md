# API Safety and Rate Limiting

**Purpose:** Documentation of the safety mechanisms implemented in the SQL ReAct Agent to prevent API suspension and handle token limit errors.

**Date:** 2025-11-30

---

## Overview

The agent implements **4 layers of protection** to ensure safe usage of the Groq API without risking account suspension.

---

## Protection Layers

### 1. **Minimum Delay Between API Calls** ⏳

**What it does:**
- Enforces a **2-second minimum delay** between consecutive LLM API calls
- Tracks the timestamp of the last API call
- Automatically sleeps if calls are made too quickly

**Why it matters:**
- Prevents triggering per-minute rate limits
- Shows respect for API resources
- Reduces suspension risk

**Implementation:**
```python
time_since_last_call = time.time() - self._last_api_call
if time_since_last_call < 2.0:  # 2-second minimum
    sleep_time = 2.0 - time_since_last_call
    time.sleep(sleep_time)
```

---

### 2. **Daily Token Limit Detection** 🛑

**What it does:**
- Detects daily token limit errors from Groq
- **Stops immediately** without retrying
- Provides clear error message with next steps

**Error keywords detected:**
- `'daily limit'`
- `'quota exceeded'`
- `'daily quota'`
- `'rate_limit_exceeded'` (Groq specific)

**What happens:**
```
🛑 DAILY TOKEN LIMIT REACHED!
======================================================================
Groq daily token limit has been exceeded.
This limit resets daily (usually at midnight UTC).

Actions:
1. Wait for limit reset (check Groq dashboard)
2. Or switch to another provider (set llm_provider='gemini')
3. Or use a different API key if available

Original error: [error details]
======================================================================
```

**Why it's critical:**
- **Prevents account suspension** from repeated failed attempts
- Saves time (no point retrying until limit resets)
- Gives clear action items

---

### 3. **Per-Minute Rate Limit Handling** 🔄

**What it does:**
- Detects per-minute rate limit errors
- **Retries with exponential backoff**
- Maximum 3 retry attempts

**Error keywords detected:**
- `'rate limit'`
- `'too many requests'`
- `'requests per minute'`

**Backoff schedule:**
- Attempt 1: Immediate
- Attempt 2: Wait 5 seconds
- Attempt 3: Wait 10 seconds
- Attempt 4: Wait 20 seconds

**Why this works:**
- Per-minute limits are temporary (reset every minute)
- Exponential backoff gives time for limit to reset
- Most per-minute issues resolve within 30 seconds

---

### 4. **General Error Retry Logic** 🔧

**What it does:**
- Retries on other types of errors (network issues, timeouts, etc.)
- Uses exponential backoff
- Maximum 3 retries

**Why it helps:**
- Handles transient network issues
- Recovers from temporary API problems
- Provides resilience

---

## Configuration

All protection parameters are configurable via `AgentConfig`:

```python
from agent import create_agent, AgentConfig

# Custom configuration
config = AgentConfig(
    min_delay_between_calls=2.0,  # 2-second minimum (recommended)
    max_retries=3,                # Maximum retry attempts
    retry_delay=5.0,              # Initial retry delay (exponential)
    verbose=True                  # Show detailed logging
)

agent = create_agent(
    db_path='database/company.db',
    api_key='your-key',
    config=config
)
```

---

## Best Practices

### For Testing:
1. **Start with single queries** (`test_single_query.py`)
2. **Monitor output** for rate limit warnings
3. **Wait between test runs** (at least 2 seconds)

### For Production:
1. **Keep default 2-second delay** (don't reduce)
2. **Monitor daily usage** via Groq dashboard
3. **Have fallback provider** ready (e.g., Gemini)
4. **Consider caching** common queries

### If You Hit Daily Limit:
1. **Stop immediately** (agent will do this automatically)
2. **Check Groq dashboard** for limit reset time
3. **Switch to Gemini** if urgent:
   ```python
   agent = create_agent(
       db_path='database/company.db',
       llm_provider='gemini'  # ← Switch provider
   )
   ```
4. **Wait for reset** (usually midnight UTC)

---

## Error Examples

### Daily Limit Error (Immediate Stop):
```
❌ API Error: Daily token limit exceeded
🛑 DAILY TOKEN LIMIT REACHED!
======================================================================
[Stops immediately, no retries]
```

### Per-Minute Limit (Retries):
```
⚠️  Rate limit hit (requests/minute)
⏳ Backing off: Waiting 5.0s before retry...
🔄 Retry attempt 2/3
```

### Network Error (Retries):
```
❌ API Error: Connection timeout
⏳ Waiting 5.0s before retry...
🔄 Retry attempt 2/3
```

---

## Token Usage Estimates

**For reference** (approximate):

| Query Type | Iterations | API Calls | Est. Tokens | Est. Cost (Groq) |
|------------|-----------|-----------|-------------|------------------|
| Simple ("Count employees") | 1-2 | 2-3 | ~1,000 | ~$0.001 |
| Medium ("Average by dept") | 2-3 | 3-4 | ~2,000 | ~$0.002 |
| Complex ("Multi-step") | 3-5 | 5-7 | ~4,000 | ~$0.004 |

**Groq Free Tier:** ~14,400 requests/day (check current limits)

**Safe usage:**
- 10-20 test queries = ~200-400 API calls = well within limits
- Monitor dashboard for actual usage

---

## Summary

🛑 **Daily limit** → Stops immediately, no retries
🔄 **Per-minute limit** → Retries with backoff
⏳ **Always** → 2-second delay between calls
✅ **Result** → Safe, respectful API usage

**Your account is protected!** The agent will never spam the API or ignore limits.

---

## Next Steps

1. Run `test_single_query.py` to verify protections work
2. Monitor output for any rate limit warnings
3. If all good, run `example_usage.py` for full tests

**Questions?** Check Groq dashboard:
- https://console.groq.com/usage
