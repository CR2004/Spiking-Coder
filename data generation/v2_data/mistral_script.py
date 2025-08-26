# This script generates natural language problem statements for raw human written Spiking Neural Network (SNN) code snippets using the Mistral API.
# It includes robust error handling, retry logic, and checkpointing to ensure progress is saved.

import os
import json
import time
from mistralai import Mistral

api_key = "API_KEY_HERE"  # Replace with your actual API key
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

INPUT_FILE = "snn_blocks_no_imports.json" # Input file with just raw code snippets parsed from GitHub. Not available in repo.
OUTPUT_FILE = "snn_blocks_with_problems.json" # Synthetically generated problem statements + human written code snippets.
CHECKPOINT_FILE = "checkpoint.json"

PROMPT_TEMPLATE = """
You are an expert SNN programmer. You are given a Spiking Neural Network (SNN) code snippet. Generate a natural, concise problem statement (2-4 sentences) that would lead someone to write this code.
Be direct and write as if giving an order.
The problem statement should:
- Sound like a realistic request someone might ask
- Include the key technical details from the code (frameworks, functionality, parameters)
- Be clear about the purpose or application
- Feel conversational and varied in phrasing
- Be concise and direct

Output ONLY the problem statement.

Code snippet:
{code}
"""

def load_snippets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_checkpoint():
    """Load existing results and last processed index"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
            return checkpoint.get("results", []), checkpoint.get("last_index", -1)
    return [], -1

def save_checkpoint(results, index):
    """Save current progress"""
    checkpoint = {
        "results": results,
        "last_index": index,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def call_mistral_with_retry(code_snippet, max_retries=3, timeout=30):
    """Call Mistral API with retry logic and timeout"""
    prompt = PROMPT_TEMPLATE.format(code=code_snippet)
    
    for attempt in range(max_retries):
        try:
            print(f"  API call attempt {attempt + 1}")
            
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            print(f"  Success")
            return result
            
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {str(e)[:100]}...")
            if attempt < max_retries - 1:
                wait_time = 20 
                print(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"  All attempts failed for this snippet")
                return None

def main():
    snippets = load_snippets(INPUT_FILE)
    results, last_index = load_checkpoint()
    
    total_snippets = len(snippets)
    start_index = last_index + 1
    
    print(f"Total snippets: {total_snippets}")
    print(f"Starting from index: {start_index}")
    print(f"Already processed: {len(results)}")
    
    # Process snippets starting from where we left off
    for i in range(start_index, total_snippets):
        snippet = snippets[i]
        code = snippet["code"]
        
        print(f"\nProcessing snippet {i + 1}/{total_snippets}")
        
        # Skip very long code snippets (likely to cause issues)
        if len(code) > 10000:
            print(f"  Skipping - code too long ({len(code)} chars)")
            continue
            
        # Skip empty or very short snippets
        if len(code.strip()) < 120:
            print(f"  Skipping - code too short ({len(code)} chars)")
            continue
        
        problem = call_mistral_with_retry(code)
        
        if problem:
            results.append({
                "prompt": problem,
                "output": code
            })
            print(f"  Added result #{len(results)}")
        else:
            print(f"  Skipped due to API failure")
        
        # Save checkpoint every 10 items
        if (i + 1) % 10 == 0:
            save_checkpoint(results, i)
            print(f"  Checkpoint saved at index {i}")
        
        # Add small delay to avoid hitting rate limits
        time.sleep(1)
    
    # Final save
    save_results(results, OUTPUT_FILE)
    save_checkpoint(results, total_snippets - 1)
    
    print(f"\nComplete! Saved {len(results)} problem statements to {OUTPUT_FILE}")
    
    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file cleaned up")

if __name__ == "__main__":
    main()