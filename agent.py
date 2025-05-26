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

        # Run the query with timeout
        logger.info("Starting query execution")
        start_time = time.time()
        try:
            logger.info("About to call agent.run()")
            result = await asyncio.wait_for(
                agent.run(
                    "Create a game Scenario for RPG game.",
                    max_steps=30,
                ),
                timeout=100  # 5 minute timeout for the entire operation
            )
            logger.info(f"Query completed in {time.time() - start_time:.2f} seconds")
            print(f"\nResult: {result}")
        except asyncio.TimeoutError:
            logger.error("Query execution timed out after 5 minutes")
            print("\nOperation timed out. Please try again later.")
        except Exception as e:
            logger.error(f"Error during query execution: {str(e)}")
            print(f"\nError: {str(e)}")

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