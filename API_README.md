# SQL Database Agent API

This API provides a REST interface to the SQL Database Agent, allowing you to query a SQLite database using natural language.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Google API key (optional, a default key is provided but may have usage limits):

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

3. Start the API server:

```bash
uvicorn api:app --reload
```

The server will start at http://localhost:8000

## API Endpoints

### Query the Database with Natural Language

```
POST /query
```

Send a natural language query to the database.

**Request Body:**

```json
{
  "query": "Show me all users who purchased an iPhone",
  "refine_response": true
}
```

- `query`: Natural language query to process
- `refine_response`: Whether to refine the response using Gemini-2.5-flash (optional, default: true)

**Response:**

```json
{
  "result": "Based on the database, here are all users who purchased an iPhone: [list of users]",
  "refined": true
}
```

### Execute Raw SQL Query

```
POST /sql?sql_query=SELECT * FROM users LIMIT 5
```

Execute a raw SQL query against the database (SELECT queries only).

**Parameters:**

- `sql_query`: SQL query to execute (must be a SELECT query)

**Response:**

```json
{
  "result": "| id | name | email |\n|---|------|-------|\n| 1 | John | john@example.com |..."
}
```

### List Database Tables

```
GET /tables
```

List all tables in the database.

**Response:**

```json
{
  "tables": ["users", "products", "orders", "user_categories"]
}
```

### Get Table Schema

```
GET /schema/{table_name}
```

Get the schema for a specific table.

**Parameters:**

- `table_name`: Name of the table

**Response:**

```json
{
  "schema": "| cid | name | type | notnull | dflt_value | pk |\n|-----|------|------|---------|------------|----|\n| 0 | id | INTEGER | 1 | | 1 |..."
}
```

### Check API Health

```
GET /health
```

Check the health of the API and its components.

**Response:**

```json
{
  "status": "ok",
  "database_connected": true,
  "llm_initialized": true,
  "agent_initialized": true
}
```

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all users who purchased an iPhone"}'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "Show me all users who purchased an iPhone"}
)

print(response.json())
```

### Using JavaScript fetch

```javascript
fetch("http://localhost:8000/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    query: "Show me all users who purchased an iPhone"
  }),
})
.then(response => response.json())
.then(data => console.log(data));
```

## Interactive Documentation

Visit http://localhost:8000/docs for interactive Swagger documentation, where you can test all endpoints directly from your browser.
