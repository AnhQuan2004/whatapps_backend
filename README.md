# WhatsApps Backend SQL Agent

This project provides a natural language interface to a SQL database, allowing users to ask questions in plain English and receive answers based on the database content. It can be run as a command-line tool or as a web API.

## Features

*   **Natural Language Queries**: Ask questions in English instead of writing SQL.
*   **LangChain and Gemini**: Uses LangChain and Google's Gemini LLM for state-of-the-art language understanding.
*   **Command-Line Interface**: A simple CLI for direct interaction.
*   **Web API**: A FastAPI-based web service for programmatic access.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created with the necessary packages: `fastapi`, `uvicorn`, `langchain`, `langchain-google-genai`, `sqlalchemy`, `pysqlite3-binary`, `langgraph`)*

2.  **Set Google API Key**:
    Make sure your Google API key is set as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

3.  **Database**:
    Place your SQLite database file (e.g., `messages.db`) in the root of the project directory.

## Running the Command-Line Tool

To ask a question from the command line, run `test.py` with your query as an argument:

```bash
python test.py "who is selling iphones?"
```

If you run it without arguments, it will prompt you to enter a query.

## Running the Web API

The web API provides an endpoint to ask questions programmatically.

1.  **Start the API Server**:
    ```bash
    uvicorn api:app --reload
    ```

2.  **Send a Query**:
    You can now send `POST` requests to the `/query` endpoint. For example, using `curl`:
    ```bash
    curl -X POST "http://127.0.0.1:8000/query" \
         -H "Content-Type: application/json" \
         -d '{"query": "who is selling iphones?"}'
    ```

    The API will return a JSON response with the answer.

3.  **API Documentation**:
    Interactive API documentation (provided by Swagger UI) is available at `http://127.0.0.1:8000/docs`.

### Testing with Postman

To test the API with Postman, follow these steps:

1.  **Start the API Server** as described above.
2.  **Open Postman** and create a new request.
3.  **Set the Request Type**: Select `POST`.
4.  **Set the Request URL**: Enter `http://127.0.0.1:8000/query`.
5.  **Set the Headers**:
    *   Go to the **Headers** tab.
    *   Add a new header with `Content-Type` as the key and `application/json` as the value.
6.  **Set the Body**:
    *   Go to the **Body** tab.
    *   Select the **raw** radio button and choose **JSON** from the dropdown.
    *   Enter your query in the following format:
        ```json
        {
            "query": "your question here"
        }
        ```
        For example:
        ```json
        {
            "query": "who is selling iphones?"
        }
        ```
7.  **Send the Request**: Click the **Send** button. You should see the JSON response from the API in the response panel.