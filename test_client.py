#!/usr/bin/env python3
"""
Simple client to test the OpenAI-compatible Ollama API
"""

import requests
import json
import sys

# API Configuration
API_BASE_URL = "http://98.92.8.6:3000"  # Your API server URL
MODEL_NAME = "llama2"  # Default model name

def test_chat_completion(question: str, model: str = MODEL_NAME, stream: bool = False):
    """Test the /v1/chat/completions endpoint"""
    url = f"{API_BASE_URL}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": stream,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"🤖 Asking: {question}")
    print(f"📡 Using model: {model}")
    print("-" * 50)
    
    try:
        if stream:
            # Streaming response
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            
            print("💬 Response (streaming):")
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # Remove 'data: ' prefix
                        if data_text == '[DONE]':
                            break
                        try:
                            data = json.loads(data_text)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            print("\n")
        else:
            # Non-streaming response
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            print("💬 Response:")
            print(answer)
            print("\n📊 Usage:")
            print(f"  Prompt tokens: {result['usage']['prompt_tokens']}")
            print(f"  Completion tokens: {result['usage']['completion_tokens']}")
            print(f"  Total tokens: {result['usage']['total_tokens']}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing response: {e}")
        return None

def test_text_completion(prompt: str, model: str = MODEL_NAME, stream: bool = False):
    """Test the /v1/completions endpoint"""
    url = f"{API_BASE_URL}/v1/completions"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"📝 Prompt: {prompt}")
    print(f"📡 Using model: {model}")
    print("-" * 50)
    
    try:
        if stream:
            # Streaming response
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            
            print("💬 Completion (streaming):")
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # Remove 'data: ' prefix
                        if data_text == '[DONE]':
                            break
                        try:
                            data = json.loads(data_text)
                            if 'choices' in data and len(data['choices']) > 0:
                                text = data['choices'][0].get('text', '')
                                if text:
                                    print(text, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            print("\n")
        else:
            # Non-streaming response
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            completion = result['choices'][0]['text']
            
            print("💬 Completion:")
            print(completion)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing response: {e}")
        return None

def list_available_models():
    """List available models from the API"""
    url = f"{API_BASE_URL}/v1/models"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        models = result['data']
        
        print("🔧 Available Models:")
        for model in models:
            print(f"  - {model['id']}")
        print()
        
        return [model['id'] for model in models]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching models: {e}")
        return []

def main():
    print("🚀 Testing OpenAI-compatible Ollama API")
    print("=" * 50)
    
    # List available models
    models = list_available_models()
    
    if not models:
        print("⚠️  No models available or API not accessible")
        return
    
    # Use the first available model
    model_to_use = models[0] if models else MODEL_NAME
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming."
    ]
    
    print(f"🧪 Testing Chat Completions with model: {model_to_use}")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Test {i}/3:")
        test_chat_completion(question, model_to_use, stream=False)
        print()
    
    # Test streaming
    print("🌊 Testing Streaming Response:")
    print("=" * 50)
    test_chat_completion("Tell me a short story about a robot.", model_to_use, stream=True)
    
    # Test text completion
    print("\n📄 Testing Text Completion:")
    print("=" * 50)
    test_text_completion("The future of artificial intelligence is", model_to_use, stream=False)

def interactive_mode():
    """Interactive mode to ask custom questions"""
    print("\n🎯 Interactive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    models = list_available_models()
    if not models:
        print("⚠️  No models available")
        return
        
    model_to_use = models[0]
    
    while True:
        try:
            question = input("\n💭 Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question:
                test_chat_completion(question, model_to_use, stream=False)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_mode()
        else:
            # Use command line argument as question
            question = " ".join(sys.argv[1:])
            models = list_available_models()
            model_to_use = models[0] if models else MODEL_NAME
            test_chat_completion(question, model_to_use, stream=False)
    else:
        main()