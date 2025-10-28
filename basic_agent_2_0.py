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

# ==================== CONFIGURATION ====================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("api-key")
MODEL = os.getenv("LLM-model")

# ==================== ENHANCED TOOL REGISTRY ====================

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
            function=self.download_file_content,
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
        
        self.register_tool(
            name="html_parser",
            function=self.parse_html_content,
            description="Parse HTML and extract specific elements"
        )
    
    def register_tool(self, name: str, function: callable, description: str):
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Handle both async and sync functions
        func = self.tools[tool_name]['function']
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
    
    # ========== ENHANCED TOOL IMPLEMENTATIONS ==========
    
    async def download_web_content(self, url: str) -> str:
        """Download content from web URL with enhanced error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Error downloading {url}: {str(e)}"
    
    async def download_file_content(self, url: str) -> str:
        """Download file content with length limiting"""
        content = await self.download_web_content(url)
        return content[:500]  # Limit content for demo
    
    def process_text(self, text: str, operation: str = "analyze") -> str:
        """Process text with various operations"""
        operations = {
            'reverse': lambda x: x[::-1],
            'lower': lambda x: x.lower(),
            'upper': lambda x: x.upper(),
            'analyze': lambda x: f"Text length: {len(x)}, Words: {len(x.split())}"
        }
        return operations.get(operation, lambda x: x)(text)
    
# In the ToolRegistry class, enhance the safe_calculate method:

    def safe_calculate(self, expression: str) -> str:
        """Enhanced calculator that handles natural language expressions"""
        try:
            # Clean the expression
            clean_expr = expression.lower()
            
            # Remove common phrases
            remove_words = ['calculate', 'what is', 'what\'s', 'compute', 'solve', 'the value of']
            for word in remove_words:
                clean_expr = clean_expr.replace(word, '')
            
            # Replace words with operators
            clean_expr = clean_expr.replace('plus', '+')
            clean_expr = clean_expr.replace('minus', '-')
            clean_expr = clean_expr.replace('times', '*')
            clean_expr = clean_expr.replace('multiplied by', '*')
            clean_expr = clean_expr.replace('divided by', '/')
            
            # Extract only safe characters
            safe_chars = r'[0-9+\-*/().]'
            math_expr = ''.join(re.findall(safe_chars, clean_expr))
            
            if not math_expr:
                return "Error: No valid mathematical expression found"
            
            # Evaluate safely
            result = eval(math_expr)
            return str(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except:
            return "Error: Invalid mathematical expression"
    
    def extract_url_from_text(self, text: str) -> str:
        """Enhanced URL extraction from text"""
        # More robust URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, text)
        
        if matches:
            # Clean the URL
            url = matches[0].rstrip('.,;!?')
            return url
        return ""
    
    def parse_html_content(self, html: str, element: str = "headings") -> str:
        """Parse HTML content and extract specific elements"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            if element == "headings":
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                return "\n".join([f"{h.name}: {h.get_text().strip()}" for h in headings])
            elif element == "links":
                links = soup.find_all('a', href=True)
                return "\n".join([f"{a.get_text().strip()}: {a['href']}" for a in links[:10]])
            else:
                return f"HTML content length: {len(html)} characters"
                
        except ImportError:
            return "BeautifulSoup not available for HTML parsing"
        except Exception as e:
            return f"HTML parsing error: {str(e)}"

# ==================== ENHANCED AGENT BRAIN ====================

class AgentBrain:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze_task(self, task: str) -> Dict[str, Any]:
        """Enhanced task analysis with better URL detection"""
        
        task_lower = task.lower()
        
        # Direct URL detection without LLM for reliability
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        has_url = bool(re.search(url_pattern, task))
        
        if has_url:
            if any(x in task_lower for x in ['heading', 'title', 'extract', 'find', 'scrape']):
                return {
                    "type": "web_scraping", 
                    "tools_needed": ["url_extractor", "web_downloader", "html_parser"],
                    "steps": ["extract_url", "download_content", "parse_html", "return_result"]
                }
            elif any(x in task_lower for x in ['download', 'file', 'content']):
                return {
                    "type": "file_processing", 
                    "tools_needed": ["url_extractor", "file_downloader"],
                    "steps": ["extract_url", "download_file", "return_result"]
                }
            else:
                return {
                    "type": "web_scraping", 
                    "tools_needed": ["url_extractor", "web_downloader"],
                    "steps": ["extract_url", "download_content", "return_result"]
                }
        elif any(x in task_lower for x in ['calculate', 'sum', 'add', 'multiply', 'math']):
            return {
                "type": "calculation", 
                "tools_needed": ["calculator"],
                "steps": ["calculate_expression", "return_result"]
            }
        elif any(x in task_lower for x in ['reverse', 'opposite']):
            return {
                "type": "text_manipulation", 
                "tools_needed": ["text_processor"],
                "steps": ["process_text", "return_result"]
            }
        else:
            return {
                "type": "general", 
                "tools_needed": [],
                "steps": ["direct_processing"]
            }

# In the TaskExecutor class, fix the calculate_expression step:

class TaskExecutor:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
    
    async def execute_workflow(self, workflow: List[str], task: str, task_analysis: Dict[str, Any]) -> str:
        """Enhanced workflow execution with better error handling"""
        
        context = {'original_task': task}
        
        for step in workflow:
            try:
                print(f"  Executing step: {step}")
                
                if step == 'extract_url':
                    context['url'] = self.tools.extract_url_from_text(task)
                    print(f"    Extracted URL: {context['url']}")
                    
                elif step == 'download_content':
                    if 'url' in context and context['url']:
                        context['content'] = await self.tools.use_tool('web_downloader', url=context['url'])
                        print(f"    Downloaded content length: {len(context['content'])}")
                        
                elif step == 'download_file':
                    if 'url' in context and context['url']:
                        context['content'] = await self.tools.use_tool('file_downloader', url=context['url'])
                        print(f"    Downloaded file content: {context['content'][:100]}...")
                        
                elif step == 'parse_html':
                    if 'content' in context:
                        element_type = "headings" if "heading" in task.lower() else "general"
                        context['result'] = await self.tools.use_tool('html_parser', html=context['content'], element=element_type)
                        
                elif step == 'process_text':
                    operation = "reverse" if "reverse" in task.lower() else "analyze"
                    context['result'] = await self.tools.use_tool('text_processor', text=task, operation=operation)
                    
                elif step == 'calculate_expression':  # FIXED THIS STEP
                    # Extract calculation part from task
                    calc_expression = self._extract_calculation(task)
                    context['result'] = await self.tools.use_tool('calculator', expression=calc_expression)
                    print(f"    Calculated: {calc_expression} = {context['result']}")
                    
                elif step == 'direct_processing':
                    context['result'] = await self._process_directly(task)
                    
                elif step == 'return_result':
                    final_result = context.get('result', 
                                context.get('content', 'Task completed but no specific result generated'))
                    return str(final_result)
                    
            except Exception as e:
                print(f"    Error in step {step}: {str(e)}")
                context['error'] = str(e)
                return f"Error in step {step}: {str(e)}"
        
        return context.get('result', 'Workflow completed without final result')
    
    def _extract_calculation(self, task: str) -> str:
        """Extract calculation expression from task text"""
        # Remove common calculation phrases
        expression = task.lower()
        remove_phrases = [
            'calculate', 'what is', 'what\'s', 'compute', 'solve', 
            'the value of', 'evaluate', 'result of'
        ]
        
        for phrase in remove_phrases:
            expression = expression.replace(phrase, '')
        
        # Extract mathematical expressions
        math_pattern = r'[0-9+\-*/().]+'
        matches = re.findall(math_pattern, expression)
        
        if matches:
            # Join matches and clean up
            calc_expr = ''.join(matches)
            # Ensure it's a valid expression
            if any(op in calc_expr for op in ['+', '-', '*', '/', '(', ')']):
                return calc_expr
        
        # If no clear expression found, try to extract numbers and operations
        return task.strip()

    async def _process_directly(self, task: str) -> str:
        """Process task directly using LLM"""
        try:
            messages = [
                {"role": "system", "content": "Solve the task directly and return only the answer."},
                {"role": "user", "content": task}
            ]
            response = call_openrouter(messages)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Direct processing failed: {str(e)}"
    async def _process_directly(self, task: str) -> str:
        """Process task directly using LLM"""
        try:
            messages = [
                {"role": "system", "content": "Solve the task directly and return only the answer."},
                {"role": "user", "content": task}
            ]
            response = call_openrouter(messages)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Direct processing failed: {str(e)}"

# ==================== REMAINING CLASSES (UNCHANGED) ====================

class TaskPlanner:
    def create_workflow(self, task_analysis: Dict[str, Any]) -> List[str]:
        workflow_templates = {
            'web_scraping': ['extract_url', 'download_content', 'parse_html', 'return_result'],
            'file_processing': ['extract_url', 'download_file', 'return_result'],
            'calculation': ['calculate_expression', 'return_result'],
            'text_manipulation': ['process_text', 'return_result'],
            'general': ['direct_processing']
        }
        return workflow_templates.get(task_analysis.get('type', 'general'), ['direct_processing'])

class AgentMemory:
    def __init__(self):
        self.conversation_history = []
    
    def remember(self, task: str, result: str):
        self.conversation_history.append({'task': task, 'result': result, 'timestamp': time.time()})
    
    def recall_similar(self, task: str) -> Optional[str]:
        for entry in self.conversation_history[-10:]:
            if task.lower() in entry['task'].lower() or entry['task'].lower() in task.lower():
                return entry['result']
        return None

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
        return call_openrouter(messages)
    
    async def process_task(self, task: str) -> Dict[str, Any]:
        cached_result = self.memory.recall_similar(task)
        if cached_result:
            return {"action": "cached", "result": cached_result}
        
        try:
            task_analysis = await self.brain.analyze_task(task)
            workflow = self.planner.create_workflow(task_analysis)
            result = await self.executor.execute_workflow(workflow, task, task_analysis)
            self.memory.remember(task, result)
            
            return {"action": f"executed_{task_analysis.get('type', 'unknown')}", "result": result}
            
        except Exception as e:
            return {"action": "error", "result": {"reason": f"Agent processing failed: {str(e)}"}}

# ==================== EXISTING FUNCTIONS ====================

def call_openrouter(messages: List[Dict[str, Any]], max_retries: int = 2) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.1}

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
    try:
        return json.loads(raw_text.strip())
    except Exception as e:
        raise ValueError(f"Assistant output is not valid JSON: {e}; raw: {raw_text!r}")

# ==================== MAIN AGENT RUNNER ====================

async def run_enhanced_agent(tasks: List[str]) -> List[Dict[str, Any]]:
    agent = EnhancedAIAgent(API_KEY, MODEL)
    results = []

    for i, task in enumerate(tasks, start=1):
        print(f"🔄 Processing task {i}: {task[:60]}...")
        
        try:
            # Use enhanced agent for all tasks now
            result = await agent.process_task(task)
            
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

# ==================== TEST THE FIX ====================

async def main():
    """Test the fixed agent with your failing tasks"""
    
    test_tasks = [
        "Extract all headings from https://en.wikipedia.org/wiki/Artificial_intelligence",
        "Download and show first 33 characters of https://sample-files.com/downloads/documents/txt/ascii-art.txt",
        "Reverse this text: Hello AI World",
        "Calculate 125 * 8 + 300"
    ]
    
    print("🚀 Starting Enhanced AI Agent with URL fixes...")
    results = await run_enhanced_agent(test_tasks)
    
    print("\n📊 Results:")
    for i, result in enumerate(results, 1):
        print(f"\nTask {i}: {result['task'][:60]}...")
        print(f"Action: {result['action']}")
        print(f"Result: {result['result'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())