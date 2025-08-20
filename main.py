try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
    print("Using pysqlite3 with extension support")
except ImportError:
    print("Using standard sqlite3 module (limited extension support)")

import os
import sys
import logging
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Database imports
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import langgraph, with a helpful error message if it fails
try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    logger.error("Required package 'langgraph' not found. Please install it with: pip install langgraph")
    print("Error: Required package 'langgraph' not found. Please install it with: pip install langgraph")
    sys.exit(1)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

db_instance = None
agent_executor = None
llm = None

def setup_database(db_path: str) -> SQLDatabase:
    """
    Set up and connect to the SQLite database.
    """
    try:
        logger.info(f"Connecting to database at {db_path}")
        
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Check if we can connect to the database
        try:
            connection = engine.connect()
            connection.close()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise Exception(f"Failed to connect to database: {e}")
        
        # Try to create SQLDatabase with options to handle missing extensions
        try:
            # Include all tables instead of just messages
            db = SQLDatabase(engine, sample_rows_in_table_info=2)
            logger.info("Database connection established with all tables")
        except Exception as inner_e:
            logger.warning(f"Failed with all tables: {inner_e}")
            # Fallback to simple connection without reflection
            db = SQLDatabase(engine, include_tables=[], sample_rows_in_table_info=0)
            logger.info("Database connection established with empty table set")
            
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def setup_llm() -> Any:
    """
    Initialize the LLM for the agent.
    """
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        logger.info("Initializing Gemini LLM")
        model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
        logger.info("LLM initialized successfully")
        return model
    except Exception as e:
        error_msg = f"Error initializing LLM: {e}"
        logger.error(error_msg)
        print(error_msg)
        raise


def get_default_sql_system_prompt() -> str:
    """
    Return a default SQL system prompt without requiring internet connection.
    
    Returns:
        A string containing the system prompt for SQL agent
    """
    return """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

IMPORTANT: NEVER return "I don't know" without first exhaustively exploring all possible tables and columns. Follow this systematic approach:

1. First, examine ALL available tables using sql_db_list_tables to get a complete overview of the database structure.

2. For each potentially relevant table, examine its schema using sql_db_schema to understand its columns and relationships.

3. For queries about specific items (products, people, etc.), check ALL tables that might contain relevant information:
   - Check the user_categories table which often contains product information in the category and reason columns
   - Check for text columns in ALL tables that might contain the search term using LIKE '%search_term%'
   - Look for relationships between tables that might connect users to products or other entities

4. If direct matches aren't found, try broader searches:
   - If searching for "iPhone", also try "Apple", "phone", "mobile", etc.
   - If searching for specific products, try looking for the product category
   - Check for partial matches or related terms

5. Always sample data from tables to understand their content before concluding no data exists:
   - Use "SELECT * FROM table_name LIMIT 5" to see sample data
   - Look for patterns in the data that might help interpret the query

6. If after checking ALL tables and trying multiple search approaches you still can't find relevant information, provide a detailed explanation of:
   - What tables and columns you checked
   - What search terms you tried
   - What the database structure contains
   - Suggestions for how the user might rephrase their query

Remember: Your goal is to extract ANY relevant information from the database that might help answer the query, even if it's not a perfect match.
"""

def create_sql_agent(db: SQLDatabase, llm: Any) -> Any:
    """
    Create a SQL agent with the given database and LLM.
    """
    try:
        logger.info("Creating SQL agent")
        
        # Create toolkit with database and LLM
        try:
            # Get available tables
            available_tables = db.get_usable_table_names()
            if not available_tables:
                logger.warning("No tables available in the database")
                print("\n⚠️ Warning: No tables available in the database.")
                print("This might be due to missing SQLite extensions (vec0) or empty database.")
            else:
                logger.info(f"Available tables: {', '.join(available_tables)}")
                print(f"Available tables: {', '.join(available_tables)}")
                
            # Create the toolkit with explicit sample rows to improve performance
            toolkit = SQLDatabaseToolkit(
                db=db, 
                llm=llm,
                sample_rows_in_table_info=3
            )
            
        except Exception as toolkit_error:
            logger.warning(f"Error creating toolkit with full database: {toolkit_error}")
            # Create a minimal toolkit with just the LLM
            from langchain_core.tools import Tool
            tools = [
                Tool(
                    name="query_error",
                    description="This tool always returns an error about database access",
                    func=lambda _: "Error: Cannot access database tables due to missing SQLite extensions."
                )
            ]
            logger.info("Created fallback toolkit")
            
        # Use local system prompt instead of pulling from LangChain Hub
        logger.info("Using local system prompt (no internet connection required)")
        from langchain_core.prompts import ChatPromptTemplate
        
        # Get default system prompt
        system_prompt = get_default_sql_system_prompt()
        
        # Create prompt template with additional examples
        examples = """
Examples:
1. Question: "who is selling iphone?"
   Thought: I need to find users associated with iPhone products. Let me check all tables systematically.
   Step 1: First, I'll list all tables to get an overview.
   SQL: SELECT name FROM sqlite_master WHERE type='table';
   Result: allowed_groups, chats, messages, product_categories, user_allowed_groups, user_categories, users
   
   Step 2: Let me check the user_categories table schema as it might contain product information.
   SQL: PRAGMA table_info(user_categories);
   Result: name, phone_number, category, reason columns found
   
   Step 3: Now I'll check if there's any iPhone data in user_categories.
   SQL: SELECT name, phone_number, category, reason FROM user_categories WHERE category = 'apple' OR reason LIKE '%iPhone%' OR reason LIKE '%phone%' LIMIT 5;
   Result: Found users selling iPhones in the reason column
   
   Step 4: Let me also check if there are any other tables with iPhone information.
   SQL: SELECT * FROM product_categories WHERE name = 'apple' OR name LIKE '%phone%' LIMIT 5;
   
   Final Answer: Based on the data in user_categories table, here are users selling iPhones: [list results]
   
2. Question: "show me all groups and members"
   Thought: I need to find relationships between groups and users by examining multiple tables.
   Step 1: First, I'll list all tables to find group-related tables.
   SQL: SELECT name FROM sqlite_master WHERE type='table';
   Result: Found allowed_groups, user_allowed_groups tables that might contain group information
   
   Step 2: Let me check the schema of these tables.
   SQL: PRAGMA table_info(allowed_groups); PRAGMA table_info(user_allowed_groups);
   Result: Found group_jid in allowed_groups and user_id, group_jid in user_allowed_groups
   
   Step 3: Now I'll check the users table to get member information.
   SQL: PRAGMA table_info(users);
   Result: Found id, email columns
   
   Step 4: Now I can join these tables to get groups and their members.
   SQL: SELECT ag.group_jid, u.email FROM allowed_groups ag JOIN user_allowed_groups uag ON ag.group_jid = uag.group_jid JOIN users u ON uag.user_id = u.id LIMIT 10;
   
3. Question: "who is selling samsung products?"
   Thought: I need to search for Samsung products across all relevant tables.
   Step 1: First, I'll list all tables.
   SQL: SELECT name FROM sqlite_master WHERE type='table';
   
   Step 2: Check product_categories for Samsung.
   SQL: SELECT * FROM product_categories WHERE name LIKE '%samsung%' LIMIT 5;
   
   Step 3: Check user_categories for Samsung.
   SQL: SELECT name, phone_number, category, reason FROM user_categories WHERE category LIKE '%samsung%' OR reason LIKE '%samsung%' LIMIT 10;
   Result: Found users selling Samsung products
   
   Step 4: Check if there are any other tables with product information.
   SQL: SELECT * FROM messages WHERE text LIKE '%samsung%' LIMIT 5;
   
   Final Answer: Based on the data in user_categories table, here are users selling Samsung products: [list results]
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n" + examples)
        ])
        system_message = prompt_template.format(dialect="SQLite", top_k=5)
        
        # Create agent
        logger.info("Creating ReAct agent")
        try:
            # Get the tools from toolkit
            tools = toolkit.get_tools()
            # Store the database in a separate variable for access in other functions
            db_instance = db
            
            # Create the agent with the tools using a more compatible approach
            try:
                # Try the newer langgraph approach first
                agent = create_react_agent(llm, tools, prompt=system_message)
                # Add the database to the agent for reference
                agent.db = db_instance
            except TypeError:
                # Fallback to older approach if needed
                logger.info("Falling back to older langgraph API")
                from langgraph.graph import StateGraph, END
                
                # Define a simple graph
                graph = StateGraph({"messages": []})
                graph.add_node("agent", lambda state: {"messages": state["messages"] + [("assistant", llm.invoke(state["messages"]))]})
                graph.add_edge("agent", END)
                graph.set_entry_point("agent")
                
                # Compile the graph
                agent = graph.compile()
                # Add the database to the agent for reference
                agent.db = db_instance
        except Exception as agent_error:
            # If we can't create the agent with toolkit, create a basic one
            logger.warning(f"Error creating agent with toolkit: {agent_error}")
            from langchain_core.tools import Tool
            tools = [
                Tool(
                    name="error_message",
                    description="Returns error about database access",
                    func=lambda _: "Error: Cannot access database due to missing SQLite extensions."
                )
            ]
            agent = create_react_agent(llm, tools, prompt="You are a helpful assistant.")

            
        logger.info("SQL agent created successfully")
        return agent
    except Exception as e:
        error_msg = f"Failed to create SQL agent: {e}"
        logger.error(error_msg)
        print(error_msg)
        raise


def verify_and_refine_response(query: str, response: str, llm: Any = None) -> str:
    """
    Verify if the response correctly answers the query and refine it if needed.
    Always uses Gemini-2.5-flash model for verification.
    """
    logger.info("Verifying and refining response using Gemini-2.5-flash")
    
    verification_prompt = f"""
**User Question:**
{query}

**Database Response:**
{response}

Based on the Database Response provided, extract the direct and relevant answer to the User Question. Remove all information that do not directly answer the question.

"""
    
    try:
        # Always use Gemini-2.5-flash for verification
        from langchain.chat_models import init_chat_model
        verification_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Use the verification model to verify and refine the response
        refined_response = verification_model.invoke(verification_prompt)
        
        # Extract just the content from the response object
        if hasattr(refined_response, 'content'):
            refined_response = refined_response.content
        
        logger.info("Response refined successfully")
        return refined_response
    except Exception as e:
        logger.error(f"Error refining response: {e}")
        # Return the original response if refinement fails
        return response


async def run_query(agent_executor: Any, query: str) -> str:
    """
    Run a natural language query against the SQL agent. 
    """
    try:
        logger.info(f"Processing query: '{query}'")
        
        final_response = ""
        try:
            events = agent_executor.stream(
                {"messages": [("user", query)]},
                stream_mode="values",
            )
            
            logger.info("Streaming agent response")
            
            for event in events:
                if "messages" in event and len(event["messages"]) > 0:
                    if hasattr(event["messages"][-1], "content"):
                        final_response = event["messages"][-1].content
                else:
                    logger.warning("Received event without messages")
            
            if final_response:
                refined_response = verify_and_refine_response(query, final_response)
                return refined_response
            else:
                return "No response from agent."

        except AttributeError as ae:
            error_msg = f"Agent error: {ae}"
            logger.error(error_msg)
            
            try:
                if hasattr(agent_executor, 'db') and agent_executor.db:
                    db = agent_executor.db
                    tables = db.get_usable_table_names()
                    
                    search_terms = []
                    query_lower = query.lower()
                    words = query_lower.split()
                    for word in words:
                        if len(word) > 3 and word not in ['who', 'what', 'when', 'where', 'show', 'list', 'find', 'get', 'selling', 'buying', 'from', 'with', 'that', 'have', 'this', 'these', 'those']:
                            search_terms.append(word)
                            
                    common_tech_terms = ['phone', 'mobile', 'tablet', 'laptop', 'computer', 'device', 'product']
                    
                    for term in common_tech_terms:
                        if term in query_lower and term not in search_terms:
                            search_terms.append(term)
                    
                    if not search_terms:
                        all_words = query_lower.split()
                        common_words = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 
                                       'show', 'list', 'find', 'get', 'selling', 'buying', 'from', 'with', 'that', 
                                       'have', 'this', 'these', 'those', 'and', 'or', 'for', 'to', 'in', 'on', 'at']
                        
                        for word in all_words:
                            if len(word) > 2 and word not in common_words:
                                search_terms.append(word)
                                
                    table_scores = {}
                    for table_name in tables:
                        score = 0
                        try:
                            schema = db.run(f"PRAGMA table_info({table_name})")
                            schema_str = str(schema)
                            
                            for term in search_terms:
                                if term.lower() in schema_str.lower():
                                    score += 5
                            
                            text_columns = []
                            for line in schema_str.split('\n'):
                                if 'TEXT' in line.upper() or 'VARCHAR' in line.upper() or 'CHAR' in line.upper():
                                    score += 1
                                    parts = line.split('|')
                                    if len(parts) > 2:
                                        column_name = parts[2].strip()
                                        text_columns.append(column_name)
                            
                            product_related_terms = ['product', 'category', 'item', 'name', 'description', 'title', 'brand']
                            for line in schema_str.split('\n'):
                                for term in product_related_terms:
                                    if term in line.lower():
                                        score += 3
                            
                            if text_columns:
                                try:
                                    sample = db.run(f"SELECT * FROM {table_name} LIMIT 3")
                                    sample_str = str(sample)
                                    
                                    for term in search_terms:
                                        if term.lower() in sample_str.lower():
                                            score += 10
                                except:
                                    pass
                            
                            table_scores[table_name] = {
                                'score': score,
                                'text_columns': text_columns
                            }
                        except Exception as e:
                            logger.error(f"Error analyzing table {table_name}: {e}")
                            
                    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                    
                    results_found = False
                    for table_name, info in sorted_tables:
                        if results_found:
                            break
                            
                        text_columns = info['text_columns']
                        if text_columns:
                            conditions = []
                            for term in search_terms:
                                for column in text_columns:
                                    conditions.append(f"{column} LIKE '%{term}%'")
                            
                            if conditions:
                                query_str = f"""
                                    SELECT *
                                    FROM {table_name}
                                    WHERE {' OR '.join(conditions)}
                                    LIMIT 10
                                """
                                
                                try:
                                    result = db.run(query_str)
                                    if str(result).strip() and not str(result).strip() == '':
                                        results_found = True
                                        return str(result)
                                except Exception as e:
                                    logger.error(f"Error querying {table_name}: {e}")
                    
                    if not results_found:
                        return "No results found after exhaustive search."
                else:
                    return "No database connection available for fallback query."
            except Exception as fallback_e:
                return f"Fallback approach failed: {fallback_e}"
        
        logger.info("Query processing completed")
        return "An unexpected error occurred."
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def find_database_file(default_path: str) -> str:
    """
    Find a suitable database file.
    """
    if os.path.exists(default_path):
        logger.info(f"Found database at default path: {default_path}")
        return default_path
    
    logger.warning(f"Database file '{default_path}' not found")
    current_dir = os.getcwd()
    
    available_files = [f for f in os.listdir(current_dir) if f.endswith('.db')]
    
    if available_files:
        selected_db = available_files[0]
        logger.info(f"Found alternative database files: {', '.join(available_files)}")
        return os.path.join(current_dir, selected_db)
    
    logger.error("No database files found in the current directory")
    raise FileNotFoundError("No database files found in the current directory")


@app.on_event("startup")
async def startup_event():
    global db_instance, agent_executor, llm
    try:
        # Load .env file from the same directory as the script
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Starting up and initializing components...")
        
        default_db_path = "messages.db"
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in .env file")
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        db_path = find_database_file(default_db_path)
        
        db_instance = setup_database(db_path)
        llm = setup_llm()
        agent_executor = create_sql_agent(db_instance, llm)
        
        logger.info("All components initialized successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}", exc_info=True)
        sys.exit(1)


@app.post("/query/")
async def query_endpoint(request: QueryRequest):
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    response = await run_query(agent_executor, request.query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
