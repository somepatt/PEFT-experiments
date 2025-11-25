from datasets import load_dataset
import torch

def format_mrpc(example):
    prompt = (
        "You are a helpful assistant that detects paraphrases.\n"
        f"Sentence 1: {example['text1']}\n"
        f"Sentence 2: {example['text2']}\n"
        "Answer with Yes or No.\nAnswer:"
    )
    label = " Yes" if example["label"] == 1 else " No"
    return {"prompt": prompt, "response": label, "label_val": example["label"]}

def format_samsum(example):
    prompt = (
        f"Summarize this dialogue:\n{example['dialogue']}\n---\n"
        f"Summary:\n"
    )
    return {"prompt": prompt, "response": example['summary']}

def get_tokenized_dataset(dataset_name, tokenizer, max_length=512):
    
    if dataset_name == "SetFit/mrpc":
        ds = load_dataset(dataset_name)
        remove_cols = ds["train"].column_names
        ds = ds.map(format_mrpc)
    elif dataset_name == "knkarthick/samsum":
        ds = load_dataset(dataset_name)
        remove_cols = ds["train"].column_names
        ds = ds.map(format_samsum)
    else:
        raise ValueError("Unknown dataset")

    def tokenize_and_mask(example):
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False 
        )["input_ids"]

        response_ids = tokenizer(
            example["response"] + tokenizer.eos_token,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + response_ids
        
        labels = [-100] * len(prompt_ids) + response_ids
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_text": example["prompt"], 
            "target_text": example["response"].strip()
        }

    tokenized_ds = ds.map(tokenize_and_mask, remove_columns=remove_cols, batched=False)
    
    return ds, tokenized_ds