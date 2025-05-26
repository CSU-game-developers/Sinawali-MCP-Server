import asyncio
import os
import json
import logging
import sys
import time
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
            logger.info("GROQ API key obtained via prompt")

        # Load configuration from JSON file
        config_path = os.path.join(os.path.dirname(__file__), 'airbnb_mcp.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info("Configuration loaded successfully")
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_path}")
            return

        # Create MCPClient from configuration
        try:
            logger.info("Creating MCP client")
            client = MCPClient.from_dict(config)
            logger.info("MCP client created successfully")
        except Exception as e:
            logger.error(f"Error creating MCP client: {str(e)}")
            return

        # Create LLM with GROQ
        try:
            logger.info("Initializing LLM with GROQ")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_retries=2,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            return

        # Create agent with the client
        logger.info("Creating MCP agent")
        agent = MCPAgent(llm=llm, client=client, max_steps=30)

        # Initial system message that defines the assistant's role
        system_message = """You are a helpful AI assistant managing a knowledge graph for a text-based RPG. 
You have access to the following tools: add_npc, update_npc, delete_npc, add_location, update_location, delete_location, 
and other tools for managing the game world. When the user provides input, first process it using your available tools 
to update the knowledge graph. Then, respond in a way that is appropriate for a text-based RPG. 
If you encounter any issues with the tools, still provide a helpful response to the user."""
        
        # Print welcome message
        print("\n=== Text-Based RPG Knowledge Graph Assistant ===")
        print("Type 'exit' or 'quit' to end the session.\n")

        # Get initial user input from terminal
        print("To begin your RPG adventure, please enter your first command:")
        user_input = input("You: ")
        
        # Check if user wants to exit immediately
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            return
        
        while True:
            # Run the query with timeout
            logger.info(f"Processing user input: {user_input}")
            start_time = time.time()
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Calling agent.run() (attempt {retry_count + 1}/{max_retries})")
                    # Combine system message with user input to provide context
                    full_prompt = f"{system_message}\n\nUser: {user_input}"
                    
                    try:
                        # Try to execute with tools
                        result = await asyncio.wait_for(
                            agent.run(
                                full_prompt,
                                max_steps=30,
                            ),
                            timeout=100  # 5 minute timeout for the entire operation
                        )
                    except Exception as tool_error:
                        # If tool usage fails, fallback to raw response
                        logger.warning(f"Tool execution failed: {str(tool_error)}. Using raw response instead.")
                        fallback_prompt = f"{system_message}\n\nUser: {user_input}\n\n(Note: Use a direct response without tools since there might be tool execution issues)"
                        result = await asyncio.wait_for(
                            llm.invoke(fallback_prompt),
                            timeout=60
                        )
                        result = result.content  # Extract content from ChatMessage
                    
                    logger.info(f"Query completed in {time.time() - start_time:.2f} seconds")
                    print(f"\nAssistant: {result}\n")
                    break  # Success, exit the retry loop
                    
                except asyncio.TimeoutError:
                    logger.error("Query execution timed out")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("\nOperation timed out after multiple attempts. Please try again with a different request.")
                    else:
                        print(f"\nOperation timed out. Retrying... (Attempt {retry_count}/{max_retries})")
                        time.sleep(1)  # Brief pause before retrying
                        
                except Exception as e:
                    logger.error(f"Error during query execution: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"\nError: {str(e)}")
                        print("The system encountered multiple errors. Let's try something else.")
                    else:
                        print(f"\nEncountered an error. Retrying... (Attempt {retry_count}/{max_retries})")
                        time.sleep(1)  # Brief pause before retrying
            
            # After max retries or successful completion, get next input
            user_input = input("You: ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for playing! Goodbye.")
                break
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted. Exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)