#!/usr/bin/env python3
"""
Simple example of calling the OpenAI-compatible Ollama API
"""

import requests
import json

# Configuration
API_URL = "http://localhost:3000/v1/chat/completions"
MODEL = "llama2"  # Change this to your available model

def ask_llm(question: str):
    """Ask a question to the LLM and get a response"""
    
    # Prepare the request payload (OpenAI format)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user", 
                "content": question
            }
        ],
        "stream": False,
        "temperature": 0.7
    }
    
    # Make the API request
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the response
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        return answer
        
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        return f"JSON Error: {e}"
    except KeyError as e:
        return f"Response format error: {e}"

def main():
    # Example questions
    questions = [
        "What is Python?",
        "Explain machine learning in one sentence.",
        "What's 2+2?",
    ]
    
    print("🤖 Testing Ollama API with simple questions")
    print("=" * 50)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = ask_llm(question)
        print(f"🤖 Answer: {answer}")
        print("-" * 30)

if __name__ == "__main__":
    main()