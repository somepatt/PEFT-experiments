from datasets import load_dataset

def format_mrpc(example, tokenizer, test=False):
    prompt = (
        "You are a helpful assistant that detects paraphrases.\n"
        f"Sentence 1: {example['text1']}\n"
        f"Sentence 2: {example['text2']}\n"
        "Answer with Yes or No.\nAnswer: "
    )
    label = "Yes" if example["label"] == 1 else "No"
    if not test:
        return {"input_text": prompt, "target_text": label, "full_text": prompt + label + tokenizer.eos_token}
    else:
        return {"input_text": prompt, "target_text": label, "full_text": prompt}


def format_samsum(example, tokenizer, test=False):
    prompt = (
        f"Summarize this dialogue:\n{example['dialogue']}\n---\n"
        f"Summary:\n"
    )
    target = example['summary']
    if not test:
        return {
            "input_text": prompt, 
            "target_text": target,
            "full_text": prompt + target + tokenizer.eos_token
        }
    else:
        return {
            "input_text": prompt, 
            "target_text": target,
            "full_text": prompt + tokenizer.eos_token
            }
        

def get_tokenized_dataset(dataset_name, tokenizer, max_length=512):
    
    if dataset_name == "SetFit/mrpc":
        ds = load_dataset(dataset_name)
        ds['train'] = ds['train'].map(lambda x: format_mrpc(x, tokenizer))
        ds['validation'] = ds['validation'].map(lambda x: format_mrpc(x, tokenizer))
        ds['test'] = ds['test'].map(lambda x: format_mrpc(x, tokenizer, test=True))
    elif dataset_name == "knkarthick/samsum":
        ds = load_dataset(dataset_name)
        ds['train'] = ds['train'].map(lambda x: format_samsum(x, tokenizer))
        ds['validation'] = ds['validation'].map(lambda x: format_samsum(x, tokenizer))
        ds['test'] = ds['test'].map(lambda x: format_samsum(x, tokenizer, test=True))
    else:
        raise ValueError("Unknown dataset")

    def tokenize_fn(examples):
        tokenizer.padding_side = "right"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer(
            examples["full_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    
    
    return ds, tokenized_ds