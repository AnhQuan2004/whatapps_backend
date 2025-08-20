
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


def run_query(agent_executor: Any, query: str) -> None:
    """
    Run a natural language query against the SQL agent. 
    """
    try:
        logger.info(f"Processing query: '{query}'")
        print(f"\nProcessing query: '{query}'")
        print("-" * 50)
        
        # For langgraph agents, tools are accessed differently
        # We don't need to check for tools availability as the agent will handle it
        
        try:
            events = agent_executor.stream(
                {"messages": [("user", query)]},
                stream_mode="values",
            )
            
            logger.info("Streaming agent response")
            final_response = ""
            for event in events:
                if "messages" in event and len(event["messages"]) > 0:
                    # Display the response
                    event["messages"][-1].pretty_print()
                    
                    # Capture the response for verification
                    if hasattr(event["messages"][-1], "content"):
                        final_response = event["messages"][-1].content
                else:
                    logger.warning("Received event without messages")
            
            # Verify and refine the response if we have one
            if final_response:
                print("\nVerifying and refining response...")
                refined_response = verify_and_refine_response(query, final_response)
                print("\n=== Refined Response ===")
                print(refined_response)
                print("========================")
        except AttributeError as ae:
            error_msg = f"Agent error: {ae}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            print("This might be due to an incompatibility with the agent structure.")
            print("Trying fallback direct query approach...\n")
            
            try:
                # Fallback to direct SQL query if possible
                if hasattr(agent_executor, 'db') and agent_executor.db:
                    db = agent_executor.db
                    # Try a simple query to list tables
                    tables = db.get_usable_table_names()
                    print(f"Available tables: {', '.join(tables)}")
                    
                    # Try to interpret the query as SQL
                    print("Attempting to interpret your query...")
                    try:
                        # For all queries, use the dynamic search approach
                        # We won't hardcode any specific query patterns
                        # Perform exhaustive search across all tables
                        print("\nPerforming exhaustive search across all tables...")
                        
                        # Step 1: Get all tables
                        tables = db.run("SELECT name FROM sqlite_master WHERE type='table'")
                        print("\nAvailable tables:")
                        print(tables)
                        
                        # Extract table names from result
                        table_names = []
                        for line in str(tables).split('\n'):
                            if line.strip() and not line.startswith('-') and not '|' in line:
                                continue
                            if '|' in line:
                                table_name = line.split('|')[1].strip()
                                if table_name and table_name != 'name':
                                    table_names.append(table_name)
                                
                        # Step 2: Extract search terms from query
                        search_terms = []
                        query_lower = query.lower()
                        # Extract potential product names or search terms
                        words = query_lower.split()
                        for word in words:
                            if len(word) > 3 and word not in ['who', 'what', 'when', 'where', 'show', 'list', 'find', 'get', 'selling', 'buying', 'from', 'with', 'that', 'have', 'this', 'these', 'those']:
                                search_terms.append(word)
                                
                        # Extract common product brands and categories dynamically
                        # Instead of hardcoding specific brands, use a more general approach
                        common_tech_terms = ['phone', 'mobile', 'tablet', 'laptop', 'computer', 'device', 'product']
                        
                        # Check if any common tech terms are in the query
                        for term in common_tech_terms:
                            if term in query_lower and term not in search_terms:
                                search_terms.append(term)
                        
                        # If no search terms found, use a generic approach
                        if not search_terms:
                            # Extract all non-common words as potential search terms
                            all_words = query_lower.split()
                            common_words = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 
                                           'show', 'list', 'find', 'get', 'selling', 'buying', 'from', 'with', 'that', 
                                           'have', 'this', 'these', 'those', 'and', 'or', 'for', 'to', 'in', 'on', 'at']
                            
                            for word in all_words:
                                if len(word) > 2 and word not in common_words:
                                    search_terms.append(word)
                                
                        print(f"\nSearch terms extracted from query: {search_terms}")
                        
                        # Step 3: Search each table for each term
                        results_found = False
                                
                        # First, analyze all tables to determine which ones might contain relevant information
                        # We'll score tables based on their column names and sample data
                        
                        table_scores = {}
                        for table_name in table_names:
                            score = 0
                            
                            # Get schema to understand the columns
                            try:
                                schema = db.run(f"PRAGMA table_info({table_name})")
                                schema_str = str(schema)
                                
                                # Score based on column names
                                for term in search_terms:
                                    if term.lower() in schema_str.lower():
                                        score += 5  # High score for column names matching search terms
                                
                                # Score based on column types (text columns are more likely to contain search terms)
                                text_columns = []
                                for line in schema_str.split('\n'):
                                    if 'TEXT' in line.upper() or 'VARCHAR' in line.upper() or 'CHAR' in line.upper():
                                        score += 1
                                        parts = line.split('|')
                                        if len(parts) > 2:
                                            column_name = parts[2].strip()
                                            text_columns.append(column_name)
                                
                                # Score based on column names that suggest product or category information
                                product_related_terms = ['product', 'category', 'item', 'name', 'description', 'title', 'brand']
                                for line in schema_str.split('\n'):
                                    for term in product_related_terms:
                                        if term in line.lower():
                                            score += 3
                                
                                # Get sample data to check content
                                if text_columns:
                                    try:
                                        sample = db.run(f"SELECT * FROM {table_name} LIMIT 3")
                                        sample_str = str(sample)
                                        
                                        # Check if sample data contains any search terms
                                        for term in search_terms:
                                            if term.lower() in sample_str.lower():
                                                score += 10  # High score if sample data contains search terms
                                    except:
                                        pass
                                
                                # Store the score and text columns
                                table_scores[table_name] = {
                                    'score': score,
                                    'text_columns': text_columns
                                }
                            except Exception as e:
                                print(f"Error analyzing table {table_name}: {e}")
                                
                                # Sort tables by score (highest first)
                                sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                                
                                print("\nTables ranked by relevance:")
                                for table_name, info in sorted_tables:
                                    print(f"- {table_name} (score: {info['score']})")
                                
                                # Search through tables in order of relevance
                                results_found = False
                                for table_name, info in sorted_tables:
                                    if results_found:
                                        break
                                        
                                    print(f"\nChecking {table_name} table (score: {info['score']})...")
                                    
                                    # Get schema again for display
                                    schema = db.run(f"PRAGMA table_info({table_name})")
                                    print("Schema:")
                                    print(schema)
                                    
                                    # Build dynamic query based on text columns and search terms
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
                                                print(f"\nResults from {table_name}:")
                                                print(result)
                                                if str(result).strip() and not str(result).strip() == '':
                                                    results_found = True
                                            except Exception as e:
                                                print(f"Error querying {table_name}: {e}")
                                
                                # If no results found yet, try more advanced techniques
                                if not results_found:
                                    print("\nTrying more advanced search techniques...")
                                    
                                    # Look for tables with potential relationships
                                    for table_name in table_names:
                                        # Skip tables we've already searched thoroughly
                                        if table_name in [t[0] for t in sorted_tables if t[1]['score'] > 0]:
                                            continue
                                            
                                        print(f"\nAnalyzing potential relationships for {table_name}...")
                                        
                                        # Get schema
                                        schema = db.run(f"PRAGMA table_info({table_name})")
                                        schema_str = str(schema)
                                        
                                        # Look for ID columns that might indicate relationships
                                        id_columns = []
                                        for line in schema_str.split('\n'):
                                            if '|' in line:
                                                parts = line.split('|')
                                                if len(parts) > 2:
                                                    column_name = parts[2].strip()
                                                    if column_name.lower().endswith('_id') or column_name.lower() == 'id':
                                                        id_columns.append(column_name)
                                        
                                        if id_columns:
                                            print(f"Found potential ID columns in {table_name}: {', '.join(id_columns)}")
                                            
                                            # Try to find relationships with other tables
                                            for id_col in id_columns:
                                                for other_table in table_names:
                                                    if other_table != table_name:
                                                        other_schema = db.run(f"PRAGMA table_info({other_table})")
                                                        other_schema_str = str(other_schema)
                                                        
                                                        # Check if this ID might be referenced in the other table
                                                        if id_col.lower() == 'id':
                                                            # This might be a primary key, look for foreign keys
                                                            table_prefix = table_name.rstrip('s')  # Simple pluralization handling
                                                            potential_fk = f"{table_prefix}_id"
                                                            
                                                            if potential_fk.lower() in other_schema_str.lower():
                                                                print(f"Found potential relationship: {table_name}.id -> {other_table}.{potential_fk}")
                                                                
                                                                # Try a join query
                                                                try:
                                                                    join_query = f"""
                                                                        SELECT t1.*, t2.*
                                                                        FROM {table_name} t1
                                                                        JOIN {other_table} t2 ON t1.id = t2.{potential_fk}
                                                                        LIMIT 5
                                                                    """
                                                                    join_result = db.run(join_query)
                                                                    print(f"\nJoin results between {table_name} and {other_table}:")
                                                                    print(join_result)
                                                                    if str(join_result).strip() and not str(join_result).strip() == '':
                                                                        results_found = True
                                                                except Exception as e:
                                                                    print(f"Error executing join query: {e}")
                                                        elif id_col.lower().endswith('_id'):
                                                            # This might be a foreign key, look for the referenced table
                                                            referenced_table = id_col.lower().replace('_id', '')
                                                            if referenced_table == other_table.lower() or f"{referenced_table}s" == other_table.lower():
                                                                print(f"Found potential relationship: {table_name}.{id_col} -> {other_table}.id")
                                                                
                                                                # Try a join query
                                                                try:
                                                                    join_query = f"""
                                                                        SELECT t1.*, t2.*
                                                                        FROM {table_name} t1
                                                                        JOIN {other_table} t2 ON t1.{id_col} = t2.id
                                                                        LIMIT 5
                                                                    """
                                                                    join_result = db.run(join_query)
                                                                    print(f"\nJoin results between {table_name} and {other_table}:")
                                                                    print(join_result)
                                                                    if str(join_result).strip() and not str(join_result).strip() == '':
                                                                        results_found = True
                                                                except Exception as e:
                                                                    print(f"Error executing join query: {e}")
                                
                                # If no results found in main tables, check all remaining tables
                                if not results_found:
                                    print("\nChecking all other tables...")
                                    for table in table_names:
                                        if table not in ['user_categories', 'product_categories', 'messages']:
                                            print(f"\nChecking {table} table...")
                                            # Get schema
                                            schema = db.run(f"PRAGMA table_info({table})")
                                            
                                            # Look for text columns to search in
                                            text_columns = []
                                            id_columns = []
                                            for line in str(schema).split('\n'):
                                                if '|' in line:
                                                    parts = line.split('|')
                                                    if len(parts) > 2:
                                                        column_name = parts[2].strip()
                                                        column_type = parts[3].strip() if len(parts) > 3 else ""
                                                        
                                                        # Identify ID columns for potential joins
                                                        if column_name.lower().endswith('_id') or column_name.lower() == 'id':
                                                            id_columns.append(column_name)
                                                        
                                                        # Identify text columns for searching
                                                        if 'TEXT' in column_type.upper() or 'VARCHAR' in column_type.upper() or 'CHAR' in column_type.upper():
                                                            text_columns.append(column_name)
                                            
                                            # Try simple search first
                                            if text_columns:
                                                # Sample data
                                                sample = db.run(f"SELECT * FROM {table} LIMIT 3")
                                                print(f"Sample data from {table}:")
                                                print(sample)
                                                
                                                # Build dynamic query for each text column
                                                for column in text_columns:
                                                    conditions = []
                                                    for term in search_terms:
                                                        conditions.append(f"{column} LIKE '%{term}%'")
                                                    
                                                    if conditions:
                                                        query_str = f"""
                                                            SELECT *
                                                            FROM {table}
                                                            WHERE {' OR '.join(conditions)}
                                                            LIMIT 5
                                                        """
                                                        
                                                        try:
                                                            result = db.run(query_str)
                                                            print(f"\nResults from {table} (column {column}):")
                                                            print(result)
                                                            if str(result).strip() and not str(result).strip() == '':
                                                                results_found = True
                                                        except Exception as e:
                                                            print(f"Error querying {table}.{column}: {e}")
                                            
                                            # Try to find potential joins with other tables
                                            if id_columns and not results_found:
                                                print(f"\nLooking for potential joins with {table}...")
                                                
                                                for id_col in id_columns:
                                                    # Find potential foreign key relationships
                                                    related_table = None
                                                    if id_col.lower() == 'id':
                                                        # This might be a primary key, look for tables referencing it
                                                        table_prefix = table.rstrip('s')  # Simple pluralization handling
                                                        for other_table in table_names:
                                                            if other_table != table:
                                                                other_schema = db.run(f"PRAGMA table_info({other_table})")
                                                                for line in str(other_schema).split('\n'):
                                                                    if f"{table_prefix}_id" in line.lower() or f"{table}_id" in line.lower():
                                                                        related_table = other_table
                                                                        related_col = f"{table_prefix}_id" if f"{table_prefix}_id" in line.lower() else f"{table}_id"
                                                                        
                                                                        print(f"Found potential relationship: {table}.id -> {other_table}.{related_col}")
                                                                        
                                                                        # Try a join query
                                                                        try:
                                                                            join_query = f"""
                                                                                SELECT t1.*, t2.*
                                                                                FROM {table} t1
                                                                                JOIN {other_table} t2 ON t1.id = t2.{related_col}
                                                                                LIMIT 5
                                                                            """
                                                                            join_result = db.run(join_query)
                                                                            print(f"\nJoin results between {table} and {other_table}:")
                                                                            print(join_result)
                                                                            if str(join_result).strip() and not str(join_result).strip() == '':
                                                                                results_found = True
                                                                        except Exception as join_e:
                                                                            print(f"Error executing join query: {join_e}")
                                                    else:
                                                        # This might be a foreign key, look for referenced table
                                                        potential_table = id_col.replace('_id', '').lower()
                                                        for other_table in table_names:
                                                            if other_table.lower() == potential_table or other_table.lower() == f"{potential_table}s":  # Simple pluralization
                                                                print(f"Found potential relationship: {table}.{id_col} -> {other_table}.id")
                                                                
                                                                # Try a join query
                                                                try:
                                                                    join_query = f"""
                                                                        SELECT t1.*, t2.*
                                                                        FROM {table} t1
                                                                        JOIN {other_table} t2 ON t1.{id_col} = t2.id
                                                                        LIMIT 5
                                                                    """
                                                                    join_result = db.run(join_query)
                                                                    print(f"\nJoin results between {table} and {other_table}:")
                                                                    print(join_result)
                                                                    if str(join_result).strip() and not str(join_result).strip() == '':
                                                                        results_found = True
                                                                except Exception as join_e:
                                                                    print(f"Error executing join query: {join_e}")
                                
                                if not results_found:
                                    fallback_response = "No results found after exhaustive search across all tables. Try refining your search terms or check if the data exists in the database."
                                    print(f"\n{fallback_response}")
                                    
                                    # Verify and refine the fallback response
                                    print("\nVerifying and refining response...")
                                    refined_response = verify_and_refine_response(query, fallback_response)
                                    print("\n=== Refined Response ===")
                                    print(refined_response)
                                    print("========================")
                                
                            except Exception as e:
                                print(f"Error during exhaustive search: {e}")
                                print("Trying simpler approach...")
                                
                                # Fallback to simpler approach
                                try:
                                    # Check if user_categories exists
                                    tables = db.run("SELECT name FROM sqlite_master WHERE type='table' AND name='user_categories'")
                                    if 'user_categories' in str(tables):
                                        result = db.run("SELECT * FROM user_categories LIMIT 5")
                                        print("\nSample data from user_categories:")
                                        print(result)
                                except Exception as inner_e:
                                    print(f"Fallback search failed: {inner_e}")
                    except Exception as sql_e:
                        print(f"SQL error: {sql_e}")
                else:
                    print("No database connection available for fallback query.")
            except Exception as fallback_e:
                print(f"Fallback approach failed: {fallback_e}")
        
        logger.info("Query processing completed")
        print("-" * 50)
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        print(f"\n❌ {error_msg}")


def get_user_query() -> str:
    """
    Get query from command line arguments or use default.
    
    """
    default_query = "who selling iphone?"
    
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        logger.info(f"Using query from command line: '{user_query}'")
        return user_query
    
    logger.info(f"No query provided, using default: '{default_query}'")
    return default_query


def find_database_file(default_path: str) -> str:
    """
    Find a suitable database file.

    """
    # Check if default path exists
    if os.path.exists(default_path):
        logger.info(f"Found database at default path: {default_path}")
        return default_path
    
    # Look for database files in current directory
    logger.warning(f"Database file '{default_path}' not found")
    print(f"Warning: Database file '{default_path}' not found")
    
    current_dir = os.getcwd()
    print(f"Looking for database files in: {current_dir}")
    
    available_files = [f for f in os.listdir() if f.endswith('.db')]
    
    if available_files:
        selected_db = available_files[0]
        logger.info(f"Found alternative database files: {', '.join(available_files)}")
        print(f"Found database files: {', '.join(available_files)}")
        print(f"Using '{selected_db}'")
        return selected_db
    
    # Check if database exists in the workspace directory
    logger.error("No database files found in the current directory")
    raise FileNotFoundError("No database files found in the current directory")


def main():
    """Main function to set up and run the SQL agent."""
    try:
        logger.info("Starting SQL Database Agent")
        print("SQL Database Agent")
        print("=" * 50)
        
        # Configuration
        default_db_path = "messages.db"
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Set API key if not already set in environment
        if not api_key:
            api_key_value = "key"
            os.environ["GOOGLE_API_KEY"] = api_key_value
            logger.warning("Using hardcoded API key")
            print("⚠️ Warning: Using hardcoded API key. Consider setting GOOGLE_API_KEY environment variable.")
        
        # Find suitable database file
        db_path = find_database_file(default_db_path)
        
        # Setup components
        print("\nInitializing components...")
        db = setup_database(db_path)
        llm = setup_llm()
        agent_executor = create_sql_agent(db, llm)
        print("✅ All components initialized successfully")
        
        # Get query from command line arguments or use default
        query = get_user_query()
        run_query(agent_executor, query)
        
        logger.info("SQL Database Agent completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
