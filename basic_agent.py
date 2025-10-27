import os
import json
import time
import requests
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from urllib.parse import urlparse
import re

load_dotenv()

# ==================== EXISTING OPENROUTER SETUP ====================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("api-key")
MODEL = os.getenv("LLM-model")

# ==================== ENHANCED AGENT ARCHITECTURE ====================

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self._register_basic_tools()
    
    def _register_basic_tools(self):
        """Register all basic tools"""
        self.register_tool(
            name="web_downloader",
            function=self.download_web_content,
            description="Download web content from URL"
        )
        
        self.register_tool(
            name="file_downloader", 
            function=self.download_file,
            description="Download file from URL"
        )
        
        self.register_tool(
            name="text_processor",
            function=self.process_text,
            description="Process text (reverse, lowercase, etc.)"
        )
        
        self.register_tool(
            name="calculator",
            function=self.safe_calculate,
            description="Perform mathematical calculations"
        )
        
        self.register_tool(
            name="url_extractor",
            function=self.extract_url_from_text,
            description="Extract URL from text"
        )
    
    def register_tool(self, name: str, function: callable, description: str):
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return await self.tools[tool_name]['function'](**kwargs)
    
    # ========== TOOL IMPLEMENTATIONS ==========
    
    async def download_web_content(self, url: str) -> str:
        """Download content from web URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error downloading {url}: {str(e)}"
    
    async def download_file(self, url: str) -> str:
        """Download file content"""
        return await self.download_web_content(url)  # Reuse for now
    
    async def process_text(self, text: str, operation: str = "analyze") -> str:
        """Process text with various operations"""
        operations = {
            'reverse': lambda x: x[::-1],
            'lower': lambda x: x.lower(),
            'upper': lambda x: x.upper(),
            'analyze': lambda x: f"Text length: {len(x)}, Words: {len(x.split())}"
        }
        return operations.get(operation, lambda x: x)(text)
    
    def safe_calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Remove dangerous characters and evaluate
            safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
            result = eval(safe_expr)
            return str(result)
        except:
            return "Error: Invalid expression"
    
    def extract_url_from_text(self, text: str) -> str:
        """Extract URL from text"""
        url_pattern = r'https?://[^\s]+'
        matches = re.findall(url_pattern, text)
        return matches[0] if matches else ""

class AgentBrain:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task and determine required tools and workflow"""
        
        analysis_prompt = f"""
        Analyze this task and determine what type it is and what tools are needed:
        TASK: {task}
        
        Available tools:
        - web_downloader: Download web content
        - file_downloader: Download files  
        - text_processor: Process text (reverse, etc.)
        - calculator: Do calculations
        - url_extractor: Extract URLs from text
        
        Respond with JSON: {{"type": "task_type", "tools_needed": ["tool1", "tool2"], "steps": ["step1", "step2"]}}
        
        Common task types: web_scraping, file_processing, calculation, text_manipulation, image_analysis
        """
        
        response = self.llm_client([
            {"role": "system", "content": "You are a task analyzer. Respond with JSON only."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        try:
            return json.loads(response)
        except:
            # Fallback analysis
            task_lower = task.lower()
            if any(x in task_lower for x in ['http', 'url', 'website']):
                return {"type": "web_scraping", "tools_needed": ["url_extractor", "web_downloader"], "steps": ["extract_url", "download_content", "extract_info"]}
            elif any(x in task_lower for x in ['calculate', 'sum', 'add', 'multiply']):
                return {"type": "calculation", "tools_needed": ["calculator"], "steps": ["calculate_expression"]}
            elif any(x in task_lower for x in ['reverse', 'opposite']):
                return {"type": "text_manipulation", "tools_needed": ["text_processor"], "steps": ["process_text"]}
            else:
                return {"type": "general", "tools_needed": [], "steps": ["direct_processing"]}

class TaskPlanner:
    def create_workflow(self, task_analysis: Dict[str, Any]) -> List[str]:
        """Create execution workflow based on task analysis"""
        
        workflow_templates = {
            'web_scraping': ['extract_url', 'download_content', 'parse_and_extract', 'return_result'],
            'file_processing': ['extract_url', 'download_file', 'process_content', 'return_result'],
            'calculation': ['parse_expression', 'calculate', 'return_result'],
            'text_manipulation': ['analyze_text', 'process_text', 'return_result'],
            'image_analysis': ['extract_url', 'analyze_image', 'return_result'],
            'general': ['direct_processing']
        }
        
        return workflow_templates.get(task_analysis.get('type', 'general'), ['direct_processing'])

class TaskExecutor:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
    
    async def execute_workflow(self, workflow: List[str], task: str, task_analysis: Dict[str, Any]) -> str:
        """Execute the workflow steps"""
        
        context = {'original_task': task}
        
        for step in workflow:
            try:
                if step == 'extract_url':
                    context['url'] = await self.tools.use_tool('url_extractor', text=task)
                elif step == 'download_content':
                    if 'url' in context:
                        context['content'] = await self.tools.use_tool('web_downloader', url=context['url'])
                elif step == 'download_file':
                    if 'url' in context:
                        context['content'] = await self.tools.use_tool('file_downloader', url=context['url'])
                elif step == 'process_text':
                    context['result'] = await self.tools.use_tool('text_processor', text=task, operation='reverse')
                elif step == 'calculate':
                    context['result'] = await self.tools.use_tool('calculator', expression=task)
                elif step == 'direct_processing':
                    # Use LLM directly for simple tasks
                    context['result'] = await self._process_directly(task)
                elif step == 'return_result':
                    return context.get('result', 'No result generated')
                    
            except Exception as e:
                context['error'] = str(e)
                return f"Error in step {step}: {str(e)}"
        
        return context.get('result', 'Workflow completed without result')
    
    async def _process_directly(self, task: str) -> str:
        """Process task directly using LLM"""
        messages = [
            {"role": "system", "content": "Solve the task directly and return only the answer."},
            {"role": "user", "content": task}
        ]
        response = call_openrouter(messages)
        return response["choices"][0]["message"]["content"].strip()

class AgentMemory:
    def __init__(self):
        self.conversation_history = []
        self.task_cache = {}
    
    def remember(self, task: str, result: str):
        self.conversation_history.append({
            'task': task,
            'result': result,
            'timestamp': time.time()
        })
    
    def recall_similar(self, task: str) -> Optional[str]:
        # Simple similarity check (in production, use embeddings)
        for entry in self.conversation_history[-10:]:  # Check last 10
            if task.lower() in entry['task'].lower() or entry['task'].lower() in task.lower():
                return entry['result']
        return None

# ==================== ENHANCED AGENT CLASS ====================

class EnhancedAIAgent:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.tools = ToolRegistry()
        self.brain = AgentBrain(self)
        self.planner = TaskPlanner()
        self.executor = TaskExecutor(self.tools)
        self.memory = AgentMemory()
    
    def __call__(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make LLM calls - compatible with existing code"""
        return call_openrouter(messages)
    
    async def process_task(self, task: str) -> Dict[str, Any]:
        """Enhanced task processing with tools and planning"""
        
        # Check memory first
        cached_result = self.memory.recall_similar(task)
        if cached_result:
            return {"action": "cached", "result": cached_result}
        
        try:
            # Analyze task
            task_analysis = await self.brain.analyze_task(task)
            
            # Create workflow
            workflow = self.planner.create_workflow(task_analysis)
            
            # Execute workflow
            result = await self.executor.execute_workflow(workflow, task, task_analysis)
            
            # Store in memory
            self.memory.remember(task, result)
            
            return {
                "action": f"executed_{task_analysis.get('type', 'unknown')}",
                "result": result
            }
            
        except Exception as e:
            return {
                "action": "error",
                "result": {"reason": f"Agent processing failed: {str(e)}"}
            }

# ==================== EXISTING FUNCTIONS (MODIFIED) ====================

def call_openrouter(messages: List[Dict[str, Any]], max_retries: int = 2) -> Dict[str, Any]:
    """Your existing function - slightly modified for reuse"""
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
    }

    attempt = 0
    while True:
        attempt += 1
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code >= 500 and attempt <= max_retries:
            time.sleep(1.0 * attempt)
            continue
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

def parse_assistant_json(raw_text: str) -> Any:
    """Your existing function"""
    try:
        parsed = json.loads(raw_text.strip())
        return parsed
    except Exception as e:
        raise ValueError(f"Assistant output is not valid JSON: {e}; raw: {raw_text!r}")

# ==================== ENHANCED MAIN AGENT RUNNER ====================

async def run_enhanced_agent(tasks: List[str]) -> List[Dict[str, Any]]:
    """Enhanced agent that can use tools and planning"""
    
    agent = EnhancedAIAgent(API_KEY, MODEL)
    results = []

    for i, task in enumerate(tasks, start=1):
        print(f"Processing task {i}: {task[:50]}...")
        
        try:
            # Use enhanced agent for complex tasks
            if any(keyword in task.lower() for keyword in ['http', 'download', 'calculate', 'reverse', 'url']):
                result = await agent.process_task(task)
            else:
                # Use simple LLM for straightforward tasks
                messages = [
                    {"role": "system", "content": "Solve the task and return only the answer."},
                    {"role": "user", "content": task}
                ]
                response = call_openrouter(messages)
                simple_result = response["choices"][0]["message"]["content"].strip()
                result = {"action": "direct_llm", "result": simple_result}
            
            results.append({
                "task": task,
                "action": result.get("action", "unknown"),
                "result": result.get("result", "No result")
            })

        except Exception as e:
            results.append({
                "task": task,
                "action": "fail",
                "result": {"reason": str(e)},
            })

    return results

# ==================== USAGE EXAMPLES ====================

async def main():
    """Demo the enhanced agent system"""
    
    test_tasks = [
        # Text manipulation
        ".tfel nruter tsuj t'nod tub rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI",
        
        # Calculation
        "What is 125 * 8 + 300?",
        
        # Web-related (will use web_downloader tool)
        "Extract the main title from https://httpbin.org/html",
        
        # File processing
        "Download and show first 100 characters of https://httpbin.org/bytes/500",
    ]
    
    print("🚀 Starting Enhanced AI Agent...")
    results = await run_enhanced_agent(test_tasks)
    
    print("\n📊 Results:")
    for i, result in enumerate(results, 1):
        print(f"\nTask {i}: {result['task'][:60]}...")
        print(f"Action: {result['action']}")
        print(f"Result: {result['result']}")

if __name__ == "__main__":
    # Run the enhanced agent
    asyncio.run(main())