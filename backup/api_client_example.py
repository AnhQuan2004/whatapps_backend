#!/usr/bin/env python3
"""
Example client for the SQL Database Agent API

This script demonstrates how to use the SQL Database Agent API
to query a SQLite database using natural language.

Usage:
    python api_client_example.py [query]
    
    If no query is provided, the script will prompt you to enter one.
"""

import sys
import requests
import json
from typing import Dict, Any, Optional

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> Dict[str, Any]:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API server is running with: uvicorn api:app --reload")
        sys.exit(1)

def list_tables() -> Dict[str, Any]:
    """List all tables in the database."""
    response = requests.get(f"{API_BASE_URL}/tables")
    response.raise_for_status()
    return response.json()

def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get the schema for a specific table."""
    response = requests.get(f"{API_BASE_URL}/schema/{table_name}")
    response.raise_for_status()
    return response.json()

def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """Execute a raw SQL query."""
    response = requests.post(f"{API_BASE_URL}/sql", params={"sql_query": sql_query})
    response.raise_for_status()
    return response.json()

def process_natural_language_query(query: str, refine_response: bool = True) -> Dict[str, Any]:
    """Process a natural language query."""
    payload = {
        "query": query,
        "refine_response": refine_response
    }
    response = requests.post(f"{API_BASE_URL}/query", json=payload)
    response.raise_for_status()
    return response.json()

def get_user_query() -> str:
    """Get query from command line arguments or prompt the user."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    else:
        try:
            query = input("Please enter your query: ")
            if not query.strip():
                print("No query provided. Exiting.")
                sys.exit(0)
            return query
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(0)

def main():
    """Main function to demonstrate API usage."""
    print("SQL Database Agent API Client Example")
    print("=" * 50)
    
    # Check if API is healthy
    health = check_api_health()
    print(f"API Health: {health['status']}")
    print(f"Database Connected: {health['database_connected']}")
    print(f"LLM Initialized: {health['llm_initialized']}")
    print(f"Agent Initialized: {health['agent_initialized']}")
    print("-" * 50)
    
    # List available tables
    tables = list_tables()
    print(f"Available tables: {', '.join(tables['tables'])}")
    print("-" * 50)
    
    # Get user query
    query = get_user_query()
    print(f"\nProcessing query: '{query}'")
    print("-" * 50)
    
    # Process the query
    try:
        result = process_natural_language_query(query)
        print("\nResult:")
        print(result['result'])
        if result['refined']:
            print("\n(Response was refined for clarity)")
    except requests.RequestException as e:
        print(f"Error processing query: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
                print(f"API Error: {error_detail}")
            except:
                print(f"Status code: {e.response.status_code}")
                print(f"Response: {e.response.text}")

if __name__ == "__main__":
    main()
