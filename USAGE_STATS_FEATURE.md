# Usage Statistics & Cost Tracking Feature

## ✅ Implementation Complete

### What Was Added:

A **comprehensive usage tracking and cost monitoring system** with a dedicated stats panel on the right side of the app.

---

## Features

### 📊 **Real-Time Usage Statistics**

#### **Per-Model Tracking:**
- **Usage Count**: Number of times each model was used
- **Input Tokens**: Total prompt tokens consumed
- **Output Tokens**: Total completion tokens generated
- **Cost**: Estimated cost per model (in USD)

#### **Session Totals:**
- **Total Queries**: Count of all API calls in session
- **Total Cost**: Sum of all costs across models

---

### 💳 **Account Balance Monitoring**

#### **OpenRouter:**
- ✅ **Credit Limit**: Your maximum credit limit
- ✅ **Used**: Amount spent so far
- ✅ **Remaining**: Available balance
- 🔄 **Refresh Button**: Fetch live data from OpenRouter API

#### **OpenAI:**
- ℹ️ OpenAI doesn't provide a balance endpoint via API
- Shows "N/A" (requires manual checking in OpenAI dashboard)

#### **LM Studio:**
- ℹ️ Shows "Local LM Studio (Free)"
- No costs (runs locally)

---

### ⚙️ **Current Configuration Display**

Shows your active settings:
- **Provider**: LM Studio / OpenAI / OpenRouter
- **Model**: Currently selected model
- **Max Tokens**: Token limit for responses
- **Temperature**: Creativity setting

---

## UI Layout

### **Two-Column Design:**

```
┌─────────────────────────────┬──────────────────┐
│  Main Chat Area (70%)       │  Stats Panel (30%)│
│                              │                  │
│  📖 Spurgeon Sermon Q&A     │  📊 Usage Stats  │
│  ─────────────────────────  │  ──────────────  │
│                              │  💳 Account      │
│  [Chat messages]             │     Balance      │
│                              │                  │
│  [User input]                │  📈 Session      │
│                              │     Stats        │
│                              │                  │
│                              │  💰 Total        │
│                              │                  │
│                              │  ⚙️ Current      │
│                              │     Config       │
└─────────────────────────────┴──────────────────┘
```

---

## Pricing Database

### **OpenAI Models** (per 1M tokens):

| Model | Input | Output |
|-------|--------|---------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |
| gpt-4 | $30.00 | $60.00 |
| gpt-3.5-turbo | $0.50 | $1.50 |
| gpt-4.1 | $5.00 | $15.00 |
| gpt-4.1-mini | $0.30 | $1.20 |

### **OpenRouter Models**:
- **Default estimate**: $1.00 input / $3.00 output per 1M tokens
- Actual prices vary by specific model

### **LM Studio**:
- **Free** (local inference)

---

## How It Works

### **1. Token Counting**

Uses `tiktoken` library (OpenAI's official tokenizer):
```python
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoding.encode(text))
```

**Fallback**: If tiktoken fails, estimates 1 token ≈ 4 characters

### **2. Cost Calculation**

```python
input_cost = (input_tokens / 1_000_000) * pricing["input"]
output_cost = (output_tokens / 1_000_000) * pricing["output"]
total_cost = input_cost + output_cost
```

### **3. Usage Recording**

Each query records:
```json
{
  "timestamp": "2025-10-03 14:32:15",
  "model": "gpt-4o",
  "provider": "OpenAI",
  "input_tokens": 1245,
  "output_tokens": 623,
  "cost": 0.009345,
  "question": "What did Spurgeon teach about..."
}
```

### **4. Data Sources**

- **Token Counts**: Extracted from API response `usage` object
- **Costs**: Calculated using pricing database
- **Balance**: Fetched from provider API (OpenRouter only)

---

## API Endpoints Used

### **OpenRouter Balance Check:**
```
GET https://openrouter.ai/api/v1/auth/key
Headers: Authorization: Bearer {api_key}

Response:
{
  "data": {
    "limit": 100.00,
    "usage": 23.45
  }
}
```

### **OpenAI:**
- ❌ No balance endpoint available via API
- Must check manually at https://platform.openai.com/usage

---

## Usage Examples

### **Scenario 1: Using OpenRouter with Qwen3-Max**

1. User asks question
2. Query uses 1,500 input tokens, 800 output tokens
3. Cost calculated: $0.0039
4. Stats panel updates:
   - **Qwen3-Max**: 1 use, $0.0039
   - **Total**: 1 query, $0.0039

### **Scenario 2: Multiple Models in Session**

- Used GPT-4o: 3 times, $0.045
- Used GPT-4o-mini: 5 times, $0.008
- **Session Total**: 8 queries, $0.053

### **Scenario 3: OpenRouter Balance**

Click "🔄 Refresh Balance":
- **Limit**: $100.00
- **Used**: $23.45
- **Remaining**: $76.55

---

## Data Persistence

### **Session-Based Storage:**
- Statistics stored in `st.session_state`
- Persists during browser session
- **Resets** when you close/refresh the page

### **Export Feature** (Future Enhancement):
- Download usage history as CSV
- Integrate with accounting systems

---

## Cost Monitoring Benefits

### **1. Budget Control**
- Track spending in real-time
- See which models are cost-effective
- Avoid surprise bills

### **2. Model Comparison**
- Compare costs across models
- Optimize model selection
- Balance cost vs. quality

### **3. Usage Analysis**
- Identify usage patterns
- Track token consumption
- Optimize prompts for efficiency

---

## Future Enhancements (Optional)

1. **CSV Export**: Download usage history
2. **Cost Alerts**: Warn when approaching budget limit
3. **Daily/Weekly Reports**: Automated summaries
4. **Per-User Tracking**: Multi-user cost allocation
5. **Model Recommendations**: Suggest cheaper alternatives
6. **Token Optimization**: Analyze prompt efficiency

---

## Technical Details

### **Dependencies Added:**
```bash
pip install tiktoken
```

### **Files Modified:**
- `app.py`: Added pricing, tracking, stats panel

### **Functions Added:**

1. **`estimate_tokens(text)`** - Count tokens in text
2. **`calculate_cost(model, input, output)`** - Calculate API cost
3. **`get_account_balance(provider, api_key)`** - Fetch balance

### **Session State Variables:**

- `usage_stats`: List of usage records
- `total_cost`: Running total of costs

---

## Example Output

### **Stats Panel Display:**

```
📊 Usage Stats
─────────────────

💳 Account Balance
[🔄 Refresh Balance]
✓ Limit: $100.00
ℹ Used: $23.45
✓ Remaining: $76.55

─────────────────

📈 Session Stats

▼ gpt-4o
   Uses: 3          Cost: $0.0450
   Input: 4,235     Output: 2,145

▼ gpt-4o-mini
   Uses: 5          Cost: $0.0080
   Input: 8,120     Output: 3,850

─────────────────

💰 Total
   Total Queries: 8
   Total Cost: $0.0530

[🗑️ Clear Stats]

─────────────────

⚙️ Current Config
Provider: OpenRouter
Model: gpt-4o
Max Tokens: 1500
Temperature: 0.7
```

---

## Summary

✅ **Real-time usage tracking** across all providers
✅ **Accurate cost calculation** with token-level precision
✅ **Live balance monitoring** for OpenRouter
✅ **Per-model statistics** for comparison
✅ **Session totals** for budget control
✅ **Clean, organized UI** with 30% stats panel

**Impact**: Full visibility into LLM costs and usage patterns, enabling informed decisions about model selection and budget management.
