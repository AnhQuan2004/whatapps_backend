#!/usr/bin/env python3
"""
SQL Database Agent API using FastAPI

This script provides a REST API interface to the SQL Database Agent.
It allows users to query a SQLite database using natural language through HTTP requests.

Usage:
    uvicorn api:app --reload

    Then visit http://localhost:8000/docs for interactive API documentation.

Example:
    curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "Show me all users who purchased an iPhone"}'
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
# Import components from main.py
from main import (
    setup_database, 
    setup_llm, 
    create_sql_agent, 
    verify_and_refine_response,
    find_database_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store initialized components
db = None
llm = None
agent_executor = None

# Define request and response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query to process")
    refine_response: bool = Field(True, description="Whether to refine the response using Gemini-2.5-flash")

class QueryResponse(BaseModel):
    result: str = Field(..., description="The answer to the query")
    refined: bool = Field(..., description="Whether the response was refined")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    database_connected: bool = Field(..., description="Whether the database is connected")
    llm_initialized: bool = Field(..., description="Whether the LLM is initialized")
    agent_initialized: bool = Field(..., description="Whether the agent is initialized")

# Startup event to initialize components
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components on startup
    initialize_components()
    yield
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down API")

def initialize_components():
    """Initialize the database, LLM, and agent components."""
    global db, llm, agent_executor
    
    try:
        logger.info("Initializing API components")
        
        # Configuration
        default_db_path = "messages.db"
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Set API key if not already set in environment
        if not api_key:
            api_key_value = os.environ.get("GOOGLE_API_KEY")
            os.environ["GOOGLE_API_KEY"] = api_key_value
            logger.warning("Using hardcoded API key")
        
        # Find suitable database file
        db_path = find_database_file(default_db_path)
        
        # Setup components
        logger.info("Setting up database connection")
        db = setup_database(db_path)
        
        logger.info("Initializing LLM")
        llm = setup_llm()
        
        logger.info("Creating SQL agent")
        agent_executor = create_sql_agent(db, llm)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise

# Create FastAPI app
app = FastAPI(
    title="SQL Database Agent API",
    description="API for querying a SQLite database using natural language",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of the API and its components."""
    return {
        "status": "ok",
        "database_connected": db is not None,
        "llm_initialized": llm is not None,
        "agent_initialized": agent_executor is not None
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a natural language query against the database.
    
    Args:
        request: QueryRequest containing the query and options
        
    Returns:
        QueryResponse with the result
        
    Raises:
        HTTPException: If components aren't initialized or query processing fails
    """
    if not all([db, llm, agent_executor]):
        raise HTTPException(status_code=503, detail="Service components not fully initialized")
    
    try:
        logger.info(f"Processing query: '{request.query}'")
        
        # Capture the agent's response
        from io import StringIO
        import sys
        
        # Redirect stdout to capture output
        original_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Process the query using the agent
            events = agent_executor.stream(
                {"messages": [("user", request.query)]},
                stream_mode="values",
            )
            
            final_response = ""
            for event in events:
                if "messages" in event and len(event["messages"]) > 0:
                    # Capture the response
                    if hasattr(event["messages"][-1], "content"):
                        final_response = event["messages"][-1].content
            
            # Refine the response if requested
            refined = False
            if request.refine_response and final_response:
                refined_response = verify_and_refine_response(request.query, final_response)
                if refined_response != final_response:
                    final_response = refined_response
                    refined = True
            
            # If we didn't get a response, check the captured output
            if not final_response:
                captured_text = captured_output.getvalue()
                if captured_text:
                    final_response = f"Raw agent output: {captured_text}"
                else:
                    final_response = "No response generated by the agent."
        
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        return {
            "result": final_response,
            "refined": refined
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

# Direct SQL endpoint (for advanced users)
@app.post("/sql", tags=["Advanced"])
async def execute_sql(sql_query: str = Query(..., description="SQL query to execute")):
    """
    Execute a raw SQL query against the database.
    
    WARNING: This endpoint is for advanced users only and should be used with caution.
    
    Args:
        sql_query: SQL query to execute
        
    Returns:
        Result of the SQL query
        
    Raises:
        HTTPException: If database isn't initialized or query execution fails
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Only allow SELECT queries for safety
        if not sql_query.strip().upper().startswith("SELECT"):
            raise HTTPException(status_code=403, detail="Only SELECT queries are allowed")
        
        result = db.run(sql_query)
        return {"result": str(result)}
    
    except Exception as e:
        error_msg = f"Error executing SQL query: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# List tables endpoint
@app.get("/tables", tags=["Database"])
async def list_tables():
    """
    List all tables in the database.
    
    Returns:
        List of table names
        
    Raises:
        HTTPException: If database isn't initialized
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        tables = db.get_usable_table_names()
        return {"tables": tables}
    
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Table schema endpoint
@app.get("/schema/{table_name}", tags=["Database"])
async def get_table_schema(table_name: str):
    """
    Get the schema for a specific table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Schema information for the table
        
    Raises:
        HTTPException: If database isn't initialized or table doesn't exist
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Check if table exists
        tables = db.get_usable_table_names()
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        schema = db.run(f"PRAGMA table_info({table_name})")
        return {"schema": str(schema)}
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error getting schema for table '{table_name}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Run the API server when executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
