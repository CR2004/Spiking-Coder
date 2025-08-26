#This script generates synthetic SNN programming problems using the LLaMA model. I used this script to create data from deepseek-6.7b-instruct as well.
#It reads seed SNN code snippets from a JSON file, crafts prompts, and uses the LLaMA model to generate problems and solutions.
#The generated problems and solutions are saved in a JSON file.

from vllm import LLM, SamplingParams
# std imports
from collections import Counter
import json
from typing import Optional

seed_dataset_id = 'seed_snippets.json'

snn_prompt_template_1 = """You are exceptionally skilled at designing challenging and insightful programming problems specifically related to Spiking Neural Networks (SNNs).

Please gain inspiration from the following random SNN code snippet to create a high-quality problem related to SNN modeling, learning rules, or simulation techniques. Be creative and practical. Present your output in two distinct sections: **Problem Description** and **Solution**.

Use the following code snippet as inspiration:
{seed}

Instructions for each section:
1. **Problem Description**: This should be **self-contained** and comprehensible to someone familiar with neural networks and PyTorch. Clearly define the problem ? for example, implementing an SNN component, simulating a neuron model, or applying an SNN learning rule. Include all necessary context, variable explanations, and setup assumptions.
2. **Solution**: Provide a **correct**, **efficient**, and **PyTorch-compatible** implementation that fully addresses the problem. Code should follow best practices for SNN modeling (e.g., using surrogate gradients if required)."""


snn_prompt_template_2 = """You are exceptionally skilled at analyzing and optimizing Spiking Neural Network (SNN) code.

Please gain inspiration from the following SNN code snippet to create a high-quality **SNN code optimization** problem. Be creative and focus on improving computational efficiency, numerical stability, or simulation speed. Present your output in two distinct sections: **Problem Description** and **Solution**.

Code snippet for inspiration:
{seed}

Instructions for each section:
1. **Problem Description**: This should be **completely self-contained**, defining a specific inefficient SNN implementation (e.g., slow loop-based STDP, redundant tensor operations, or inefficient neuron update). Clearly include the code to be optimized and describe why it is inefficient.
2. **Solution**: Provide a **correct**, **optimized**, and well-commented version of the original code. Explain how your changes improve performance or clarity without changing the functional behavior of the SNN."""


snn_prompt_template_3 = """You are exceptionally skilled at translating Spiking Neural Network (SNN) code across frameworks or execution models.

Please gain inspiration from the following SNN code snippet to create a **code translation problem** ? for example, translating from native PyTorch to `snnTorch`, or from a custom neuron model to a framework-specific one (like BindsNET, Brian2, or Norse). Present your output in two distinct sections: **Problem Description** and **Solution**.

Code snippet for inspiration:
{seed}

Instructions for each section:
1. **Problem Description**: Clearly describe the current format and desired translation. Include the original code snippet and context (e.g., ?this PyTorch LIF model needs to be translated into a snnTorch equivalent?). Be self-contained and specify assumptions.
2. **Solution**: Provide a **correct**, translated implementation that works in the target framework or model. Maintain equivalent functionality and explain any differences in behavior or APIs."""

results=[]
model_id = "Llama-3.1-8B-Instruct"
llm = LLM(model_id,gpu_memory_utilization=0.90,tensor_parallel_size=1,max_model_len=49056)
print ("model loaded")
tokenizer = llm.get_tokenizer()
def postprocess(input_text: str) -> str:
    """ Postprocess the model output to return the text from each section.
        This is accomplished by finding lines that contain each section header.
    """
    lines = input_text.splitlines()
    problem_keyword = "**Problem Description**" 
    solution_keyword = "**Solution**"
    
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "## Problem Description" 
        solution_keyword = "## Solution"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "# Problem Description" 
        solution_keyword = "# Solution"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "Problem Description:" 
        solution_keyword = "Solution:"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "**Problem Description:**" 
        solution_keyword = "**Solution:**"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "**Problem Description:" 
        solution_keyword = "**Solution:**"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "**Problem Description:" 
        solution_keyword = "**Solution:"
        
    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        problem_keyword = "**Problem Description:**" 
        solution_keyword = "**Solution:"

    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        raise ValueError(f"All sections not present")

    # Find the starting index of each section
    problem_start = input_text.find(problem_keyword) + len(problem_keyword)
    solution_start = input_text.find(solution_keyword) + len(solution_keyword)

    # Extract the sections
    problem_description = input_text[problem_start:solution_start - len(solution_keyword)].strip()
    solution = input_text[solution_start:].strip()
    return problem_description, solution
def format_prompt(prompt):
    return tokenizer.apply_chat_template(
        [prompt],  
        add_generation_prompt=True,
        tokenize=False
    )

def generate_output(prompts,seeds) -> str:
    outputs = llm.generate([format_prompt(p) for p in prompts],
                       SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=4096,
        stop_token_ids=[tokenizer.eos_token_id],
        )
    )
    s = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        print (generated_text)
        try:
          problem_statement, solution = postprocess(generated_text)
          results.append({
            "seed": seeds[s],
            "problem statement": problem_statement,
            "solution": solution,
            "model": "llama-instruct"
          })
        except Exception as e:
             print(f"Error:{e}")
             continue
        s+=1
          
        
    return "done"   

with open(seed_dataset_id, 'r') as file:
        seed_dataset = json.load(file)
i = 0
total=0
prompts=[]
seeds = []
for seed_id,element in seed_dataset.items():
     i += 1
     print(i)
     if(i == 100000):
       break
     if(i > 50000):
       seed = element['code']
       try:
         if(i % 3 == 0):
             prompt = {"role": "user", "content":snn_prompt_template_3.format(seed=seed)}
             prompts.append(prompt)
             seeds.append(seed)
         elif(i % 2 == 0):
             prompt = {"role": "user", "content":snn_prompt_template_2.format(seed=seed)}
             prompts.append(prompt)
             seeds.append(seed)
         else:
             prompt = {"role": "user", "content":snn_prompt_template_1.format(seed=seed)}
             prompts.append(prompt)
             seeds.append(seed)
       except Exception as e:
             print(f"Error:{e}")
             continue
    
       if(i % 10 == 0):
             generated_text = generate_output(prompts,seeds)
             prompts=[]
             seeds=[]
       if(i % 100 == 0):
             with open('llama-outputs.json', 'w') as fp:
               json.dump(results, fp)
        
with open('llama-outputs.json', 'w') as fp:
    json.dump(results, fp)

