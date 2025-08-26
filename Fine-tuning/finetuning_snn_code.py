# This script is used to fine-tune a Llama-3.2-3B model on SNN code data. I used Huggingface trainer API for this purpose.
# This was used to finetune both v1 and v2 models. The only difference is the data file used and how the fine-tuning was carried out.
# The v1 model was sequentially finetuned first on 50k raw code samples and then 100k synthetic samples.
#The v2 model was finetuned was finetuned once on 40K synthetic prompt-raw human written code pairs.

import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq
)

# CONFIGURATION 
MODEL_NAME = "llama3.2_3B"
DATA_PATH = "snn_blocks_with_problems.json"  #JSON file with data samples
OUTPUT_DIR = "./llama3.2_snn_v2_ckpt"

# CUSTOM LOGGING CALLBACK 
class CustomLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        elapsed = (time.time() - self.start_time) * 1000  # ms
        iteration = state.global_step
        total_iters = args.max_steps or int(state.max_steps or 0)
        learning_rate = logs.get("learning_rate")
        loss = logs.get("loss")

        curr_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        grad_norm = self.get_grad_norm(kwargs.get("model"))

        log_string = '> global batch {:8d}/{:8d} |'.format(iteration, total_iters)
        log_string += ' elapsed time per global batch (ms): {:.1f} |'.format(elapsed)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' loss: {:.5f} |'.format(loss)
        log_string += ' memory used by tensors {:.3f} GB (peak {:.3f} GB) |'.format(curr_mem, peak_mem)
        log_string += f' grad norm: {grad_norm:.5f}'

        print(log_string)

    def get_grad_norm(self, model):
        try:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return (total_norm ** 0.5)
        except:
            return 0.0

# LOAD TOKENIZER & MODEL 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# LOAD & PREPROCESS DATA
dataset = load_dataset("json", data_files=DATA_PATH, split="train")                                              

def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result
# We mask prompts here so that only the output part contributes to loss.
def get_tokenizer_mapping_fn(tokenizer, cutoff_len, train_on_inputs=False, add_eos_token=True):
    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["prompt"] + data_point["output"]
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, add_eos_token)

        if not train_on_inputs:
            prompt_only = data_point["prompt"]
            tokenized_prompt_only = tokenize(prompt_only, tokenizer, cutoff_len, add_eos_token)
            prompt_len = len(tokenized_prompt_only["input_ids"])
            if add_eos_token:
                prompt_len -= 1  # ignore <eos> in mask
            tokenized_full_prompt["labels"] = [-100] * prompt_len + tokenized_full_prompt["labels"][prompt_len:]

        return tokenized_full_prompt

    return generate_and_tokenize_prompt
    
tokenized_dataset = dataset.map(
    get_tokenizer_mapping_fn(tokenizer, cutoff_len=2048, train_on_inputs=False),
    remove_columns=dataset.column_names
)



for i in range(2):
    print(f"--- Example {i} ---")
    print("Input IDs:", tokenized_dataset[i]["input_ids"])
    print("Labels   :", tokenized_dataset[i]["labels"])
    print("Decoded Input:", tokenizer.decode(tokenized_dataset[i]["input_ids"]))
    print("Decoded Labels:", tokenizer.decode([
        token_id if token_id != -100 else tokenizer.pad_token_id 
        for token_id in tokenized_dataset[i]["labels"]
    ]))
    print(len(tokenized_dataset[i]["input_ids"]))
    print(len(tokenized_dataset[i]["labels"]))
    
    

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,         
    return_tensors="pt"   # ensures tensor output
)     

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-5,
    bf16=True,  
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    report_to="none"
)

# SETUP TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# START TRAINING
trainer.train()
