import os
import json
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("api-key")
MODEL = os.getenv("LLM-model")

SYSTEM_PROMPT = (
    "You are a strict task-execution assistant. For each user task you receive, you MUST reply with a single valid JSON object and NOTHING ELSE.\n"
    "The JSON must be on a single line and follow this exact schema:\n"
    "{\"action\": \"<short action label - one sentence>\", \"result\": <any valid JSON value> }\n"
    "Rules (enforce them without exception):\n"
    "1) Do not ask clarifying questions. Do not produce any text other than the exact JSON object.\n"
    "2) Do not perform or suggest actions outside the requested task. Do not explain, do not apologize.\n"
    "3) If the task cannot be completed, return {\"action\": \"fail\", \"result\": {\"reason\": \"brief reason\"}}\n"
    "4) Keep 'action' short (max 12 words). 'result' can be string, number, object, list — but must be valid JSON.\n"
)


def call_openrouter(messages: List[Dict[str, Any]], max_retries: int = 2) -> Dict[str, Any]:
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
    try:
        parsed = json.loads(raw_text.strip())
        return parsed
    except Exception as e:
        raise ValueError(f"Assistant output is not valid JSON: {e}; raw: {raw_text!r}")


def run_minimal_agent(tasks: List[str]) -> List[Dict[str, Any]]:
    results = []

    for i, task in enumerate(tasks, start=1):
        user_message = {
            "role": "user",
            "content": f"Task {i}: {task}\n\nRespond only with the required JSON object as specified by the system prompt.",
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ]

        try:
            response = call_openrouter(messages)
            content = None
            try:
                content = response["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError(f"Unexpected response shape: {response}")

            parsed = parse_assistant_json(content)

            if not isinstance(parsed, dict) or "action" not in parsed or "result" not in parsed:
                raise ValueError(f"Parsed JSON does not match required schema: {parsed}")

            results.append({"task": task, "action": parsed["action"], "result": parsed["result"]})

        except Exception as e:
            results.append({
                "task": task,
                "action": "fail",
                "result": {"reason": str(e)},
            })

    return results


if __name__ == "__main__":
    sample_tasks = [
        ".tfel nruter tsuj t'nod tub rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"
    ]

    out = run_minimal_agent(sample_tasks)
    print(json.dumps(out, indent=2))
