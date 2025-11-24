import asyncio
import os
import json
import logging
import sys
import time
import getpass
import subprocess
from typing import Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store client and agent
mcp_client = None
mcp_agent = None
mcp_server_process = None  # Track the server process
connection_state = {"connected": False, "last_checked": 0, "retry_count": 0, "max_retries": 3}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        logger.info("Starting up application...")
        await initialize_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent on startup: {str(e)}")
        # Don't fail startup, the agent will be initialized on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    global mcp_server_process
    if mcp_server_process and mcp_server_process.poll() is None:
        logger.info("Stopping MCP server process...")
        try:
            mcp_server_process.terminate()
            mcp_server_process.wait(timeout=5)
        except:
            mcp_server_process.kill()

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="MCP-USE API",
    description="API for interacting with an MCP agent for text-based RPG knowledge graph management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class QueryRequest(BaseModel):
    query: str
    system_message: Optional[str] = None
    max_steps: Optional[int] = 3  # Reduced default from 5 to 3 for faster response
    timeout: Optional[int] = 15  # Reduced default from 30 to 15 seconds

class StatusResponse(BaseModel):
    status: str
    message: str

async def start_mcp_server(server_name: str, server_config: dict) -> Optional[subprocess.Popen]:
    """Start an MCP server process if needed.
    
    Args:
        server_name: Name of the server
        server_config: Server configuration with command and args
        
    Returns:
        The process object if started, None if already running or no command provided
    """
    if not server_config.get("command"):
        logger.warning(f"No command specified for server {server_name}, assuming external process")
        return None
        
    # Build command with arguments
    cmd = [server_config["command"]] + server_config.get("args", [])
    logger.info(f"Starting MCP server '{server_name}': {' '.join(cmd)}")
    
    # Try to start the process
    try:
        # Start the process with proper redirection
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True  # Using shell=True for Windows compatibility
        )
        
        # Allow some time for the process to start
        await asyncio.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"MCP server '{server_name}' started successfully (PID: {process.pid})")
            return process
        else:
            stderr = process.stderr.read() if process.stderr else "No error output"
            logger.error(f"MCP server '{server_name}' failed to start: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error starting MCP server '{server_name}': {str(e)}")
        return None

async def verify_connection() -> Tuple[bool, str]:
    """Verify that the MCP client connection is working"""
    global mcp_client, connection_state
    
    # Only check connection every 30 seconds to avoid too many checks
    current_time = time.time()
    if current_time - connection_state["last_checked"] < 30 and connection_state["connected"]:
        return True, "Using cached connection state"
    
    connection_state["last_checked"] = current_time
    
    if mcp_client is None:
        return False, "MCP client not initialized"
    
    try:
        # Quick check: just verify we have server names configured
        server_names = mcp_client.get_server_names()
        if not server_names:
            return False, "No servers configured"
        
        # Check if we have any active sessions (fast check)
        try:
            active_sessions = mcp_client.get_all_active_sessions()
            if active_sessions:
                connection_state["connected"] = True
                connection_state["retry_count"] = 0
                return True, f"Active session found"
        except Exception as e:
            logger.debug(f"Could not get active sessions: {e}")
        
        # Don't create a test session here - it's too slow
        # Instead, we'll let the actual query attempt to use the tools
        # and fall back to direct LLM if it fails
        connection_state["connected"] = True  # Assume connected for now
        connection_state["retry_count"] = 0
        return True, "Server names available (optimistic)"
            
    except Exception as e:
        connection_state["retry_count"] += 1
        logger.error(f"Connection verification error: {str(e)}")
        return False, f"Connection failed: {str(e)} (attempt {connection_state['retry_count']})"

async def initialize_agent():
    """Initialize the MCP agent and client"""
    global mcp_client, mcp_agent, connection_state, mcp_server_process
    
    # Load environment variables if not already loaded
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
        raise HTTPException(status_code=500, detail=f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file: {config_path}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in config file: {config_path}")

    # Create MCPClient from configuration
    try:
        logger.info("Creating MCP client")
        
        # Start MCP servers if needed
        for server_name, server_config in config.get("mcpServers", {}).items():
            # Try to start server process if not already running
            if mcp_server_process is None or mcp_server_process.poll() is not None:
                mcp_server_process = await start_mcp_server(server_name, server_config)
        
        # Create the MCP client with the config
        mcp_client = MCPClient.from_dict(config)
        logger.info("MCP client created successfully")
        
        # Verify the connection to MCP with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Verifying MCP connection (attempt {attempt}/{max_retries})")
            is_connected, message = await verify_connection()
            
            if is_connected:
                logger.info(f"MCP connection verified: {message}")
                connection_state["connected"] = True
                break
            elif attempt < max_retries:
                logger.warning(f"MCP connection issue: {message}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.warning(f"Failed to verify MCP connection after {max_retries} attempts: {message}")
                logger.warning("Continuing with initialization, but MCP tools may not be available")
            
    except Exception as e:
        logger.error(f"Error creating MCP client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating MCP client: {str(e)}")

    # Create LLM with GROQ
    try:
        logger.info("Initializing LLM with GROQ")
        llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.0,
            max_retries=2,
            timeout=60,  # Add explicit timeout
        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing LLM: {str(e)}")

    # Create agent with the client
    logger.info("Creating MCP agent")
    mcp_agent = MCPAgent(llm=llm, client=mcp_client, max_steps=3)  # Reduced to 3 for faster responses
    
    return mcp_agent

def get_agent():
    """Dependency to ensure agent is initialized"""
    if mcp_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")
    return mcp_agent

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    return StatusResponse(status="ok", message="Server is running")

@app.get("/status", response_model=StatusResponse)
async def status():
    """Check if the agent is initialized"""
    if mcp_agent is None:
        return StatusResponse(status="initializing", message="Agent not yet initialized")
    return StatusResponse(status="ready", message="Agent is ready")

@app.post("/query")
async def query(request: QueryRequest, agent: MCPAgent = Depends(get_agent)):
    """Process a query using the MCP agent with aggressive timeout protection"""
    try:
        logger.info(f"Processing user input: {request.query}")
        start_time = time.time()
        
        # Use configurable values from request or defaults
        max_steps = request.max_steps or 3  # Further reduced from 5 to 3
        timeout = request.timeout or 15  # Further reduced from 30 to 15 seconds
        
        # Use provided system message or default
        system_message = request.system_message or """You are a helpful AI assistant managing a knowledge graph for a text-based RPG. 
You have access to tools for managing the game world. Process the user's input and respond concisely with a "Final Answer:"."""
        
        full_prompt = f"{system_message}\n\nUser: {request.query}"
        
        # Check connection state before running query (fast check only)
        is_connected, message = await verify_connection()
        connection_warning = ""
        
        # Try with tools first, but with very aggressive timeout
        result = None
        used_fallback = False
        
        if is_connected:
            try:
                logger.info(f"Attempting agent.run() with max_steps={max_steps}, timeout={timeout}s")
                # Wrap in asyncio.wait_for with aggressive timeout
                result = await asyncio.wait_for(
                    agent.run(full_prompt, max_steps=max_steps),
                    timeout=timeout
                )
                logger.info(f"Agent.run() completed successfully")
                
            except asyncio.TimeoutError:
                logger.warning(f"Agent.run() timed out after {timeout}s, falling back to direct LLM")
                used_fallback = True
                connection_warning = f"Tool execution timed out. Using direct response."
                
            except Exception as tool_error:
                error_msg = str(tool_error)[:200]  # Truncate long errors
                logger.warning(f"Tool execution failed: {error_msg}. Using direct response.")
                used_fallback = True
                connection_warning = f"Tool execution failed: {error_msg}"
        else:
            logger.warning(f"Connection not verified: {message}. Using direct LLM response.")
            used_fallback = True
            connection_warning = f"MCP connection issue: {message}"
        
        # If tools failed or timed out, use direct LLM
        if result is None or used_fallback:
            try:
                logger.info("Using direct LLM fallback (no tools)")
                fallback_prompt = f"{system_message}\n\nUser: {request.query}\n\n(Provide a direct response)"
                # Give LLM a shorter timeout too
                response = await asyncio.wait_for(
                    asyncio.to_thread(agent.llm.invoke, fallback_prompt),
                    timeout=10
                )
                result = response.content
                logger.info("Direct LLM response completed")
            except asyncio.TimeoutError:
                logger.error("Even direct LLM timed out!")
                raise HTTPException(
                    status_code=504,
                    detail="Request timed out. The LLM service may be slow. Try /query-fast instead."
                )
        
        processing_time = time.time() - start_time
        logger.info(f"Query completed in {processing_time:.2f} seconds")
        
        # Parse and structure the response if possible
        structured_response = {}
        
        # Try to extract structured elements from agent's response
        try:
            if isinstance(result, dict) and "steps" in result:
                # If agent result already contains structured trace data
                thoughts = []
                actions = []
                observations = []
                
                # Extract information from steps
                for step in result.get("steps", []):
                    if "thought" in step:
                        thoughts.append(step["thought"])
                    if "action" in step:
                        actions.append(step["action"])
                    if "observation" in step:
                        observations.append(step["observation"])
                
                structured_response = {
                    "thoughts": thoughts,
                    "actions": actions,
                    "observations": observations,
                    "final_answer": result.get("output", "")
                }
            elif isinstance(result, str):
                # Try to parse from string format with sections
                sections = {
                    "Thought": [],
                    "Action": [], 
                    "Observation": [],
                    "Final Answer": ""
                }
                
                # Simple parsing based on section headers
                current_section = None
                for line in result.split('\n'):
                    line = line.strip()
                    if line.startswith("Thought:"):
                        current_section = "Thought"
                        sections["Thought"].append(line[8:].strip())
                    elif line.startswith("Action:"):
                        current_section = "Action"
                        sections["Action"].append(line[7:].strip())
                    elif line.startswith("Observation:"):
                        current_section = "Observation"
                        sections["Observation"].append(line[12:].strip())
                    elif line.startswith("Final Answer:"):
                        current_section = "Final Answer"
                        sections["Final Answer"] = line[13:].strip()
                    elif current_section and line:
                        if current_section == "Final Answer":
                            sections[current_section] += " " + line
                        elif sections[current_section]:
                            sections[current_section][-1] += " " + line
                
                structured_response = sections
        except Exception as parse_error:
            logger.warning(f"Failed to parse structured response: {str(parse_error)}")
        
        # Build response
        response_data = {
            "raw_response": result,
            "structured_response": structured_response,
            "processing_time": processing_time,
            "max_steps_used": max_steps,
            "timeout_used": timeout,
            "used_fallback": used_fallback
        }
        
        if connection_warning:
            response_data["warning"] = connection_warning
            
        return response_data
                
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query-fast")
async def query_fast(request: QueryRequest, agent: MCPAgent = Depends(get_agent)):
    """Process a query using direct LLM without tools for faster response"""
    try:
        logger.info(f"Processing fast query: {request.query}")
        start_time = time.time()
        
        # Use provided system message or default
        system_message = request.system_message or """You are a helpful AI assistant for a text-based RPG. 
Provide quick, concise responses to user queries."""
        
        full_prompt = f"{system_message}\n\nUser: {request.query}"
        
        # Use direct LLM call without tools
        response = agent.llm.invoke(full_prompt)
        result = response.content  # Extract content from ChatMessage
        
        processing_time = time.time() - start_time
        logger.info(f"Fast query completed in {processing_time:.2f} seconds")
        
        return {
            "response": result,
            "processing_time": processing_time,
            "mode": "fast"
        }
                
    except Exception as e:
        logger.error(f"Error during fast query execution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/check-connection")
async def check_connection():
    """Check if the MCP connection is working"""
    global mcp_client, mcp_server_process
    
    if mcp_client is None:
        return {"status": "not_initialized", "message": "MCP client not yet initialized"}
    
    # Check if server process is running if we have one
    server_status = "unknown"
    if mcp_server_process is not None:
        exit_code = mcp_server_process.poll()
        if exit_code is None:
            server_status = "running"
        else:
            server_status = f"stopped (exit code: {exit_code})"
    
    is_connected, message = await verify_connection()
    return {
        "status": "connected" if is_connected else "disconnected",
        "message": message,
        "server_status": server_status,
        "last_checked": connection_state["last_checked"],
        "retry_count": connection_state["retry_count"]
    }

@app.post("/restart-server")
async def restart_server():
    """Attempt to restart the MCP server process"""
    global mcp_server_process, mcp_client, connection_state
    
    # Load config to get server info
    config_path = os.path.join(os.path.dirname(__file__), 'airbnb_mcp.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load config: {str(e)}"}
    
    # Kill existing process if any
    if mcp_server_process is not None:
        try:
            if mcp_server_process.poll() is None:  # Process is still running
                logger.info("Terminating existing MCP server process")
                mcp_server_process.terminate()
                try:
                    mcp_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("MCP server process did not terminate gracefully, killing it")
                    mcp_server_process.kill()
        except Exception as e:
            logger.error(f"Error terminating MCP server process: {str(e)}")
    
    # Start MCP servers from config
    server_started = False
    for server_name, server_config in config.get("mcpServers", {}).items():
        mcp_server_process = await start_mcp_server(server_name, server_config)
        if mcp_server_process is not None:
            server_started = True
    
    # Reset connection state
    connection_state["connected"] = False
    connection_state["last_checked"] = 0
    connection_state["retry_count"] = 0
    
    # Verify connection
    await asyncio.sleep(5)  # Give the server time to start up
    is_connected, message = await verify_connection()
    
    return {
        "status": "success" if server_started else "error",
        "message": "Server restarted successfully" if server_started else "Failed to restart server",
        "connection_status": "connected" if is_connected else "disconnected",
        "connection_message": message
    }

if __name__ == "__main__":
    try:
        # Check if the MCP server is running or needs to be started
        config_path = os.path.join(os.path.dirname(__file__), 'airbnb_mcp.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Print informative message about the server configuration
            print("MCP Server Configuration:")
            for server_name, server_config in config.get("mcpServers", {}).items():
                cmd_str = server_config.get("command", "")
                args_str = ' '.join(server_config.get("args", []))
                print(f"  - {server_name}: {cmd_str} {args_str}")
                
                # Validate command existence
                if cmd_str and not any(os.path.exists(os.path.join(path, cmd_str)) 
                                     for path in os.environ["PATH"].split(os.pathsep)):
                    print(f"    WARNING: Command '{cmd_str}' may not be in PATH")
                
            # Check for connection information
            if "connectionInfo" in config:
                print(f"Connection info found: {config['connectionInfo']}")
                
                # Validate connection URL
                conn_url = config.get("connectionInfo", {}).get("url", "")
                if conn_url:
                    print(f"  Connection URL: {conn_url}")
                else:
                    print("  WARNING: No connection URL specified")
            else:
                print("WARNING: No connection information found in config. Make sure the server is running.")
                
        except Exception as e:
            print(f"Error reading config: {str(e)}")
        
        # Run the FastAPI app using uvicorn
        print("Starting API server on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)  # Setting reload=False for better process control
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted. Exiting...")
        
        # Clean up server process if running
        if mcp_server_process and mcp_server_process.poll() is None:
            print("Stopping MCP server process...")
            try:
                mcp_server_process.terminate()
                mcp_server_process.wait(timeout=5)
            except:
                mcp_server_process.kill()
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)