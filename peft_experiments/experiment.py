import time
import torch
import math
import numpy as np
import evaluate
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, pipeline,
    DataCollatorForSeq2Seq
)
from transformers.pipelines.pt_utils import KeyDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import VRAMTracker, cleanup
from tqdm.auto import tqdm

class ExperimentRunner:
    def __init__(self, model_id, output_dir="./output"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def train(self, method, train_dataset, eval_dataset, batch_size=4, epochs=1):
        print(f"\n=== Training {method} ===")
        cleanup()
        
        bnb_config = None
        if method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_dora=(method == "dora")
        )

        model = get_peft_model(model, peft_config)

        self.tokenizer.padding_side = "right"
        
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{method}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=epochs,
            logging_steps=10,
            fp16=False,
            bf16=True,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=True
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8 
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        with VRAMTracker() as vram:
            start_time = time.time()
            trainer.train()
            train_time = time.time() - start_time

        adapter_path = f"{self.output_dir}/{method}/final_adapter"
        trainer.save_model(adapter_path)
        
        return {
            "method": method,
            "train_time_sec": train_time,
            "peak_vram_gb": vram.peak_memory,
            "adapter_path": adapter_path
        }

    def evaluate_classification(self, adapter_path, dataset, batch_size=8):
        """Для MRPC: Accuracy"""
        cleanup()
        print("Evaluating Classification (Accuracy)...")
        
        self.tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.load_adapter(adapter_path)
        
        pipe = pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer,
            device_map="auto", return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        correct = 0
        total = 0
        times = []

        for out, target in tqdm(zip(pipe(KeyDataset(dataset, "input_text"), batch_size=batch_size, max_new_tokens=1, do_sample=False), dataset["target_text"]), total=len(dataset)):
            
            start_t = time.time()
            generated = out[0]['generated_text'].strip().lower()
            times.append(time.time() - start_t)

            target_clean = target.strip().lower()
            
            pred_yes = generated.startswith("yes")
            target_yes = (target_clean == "yes")
            correct += int(pred_yes == target_yes)
            total += 1

        return {
            "accuracy": correct / total,
            "inference_latency_ms": (np.mean(times) / batch_size) * 1000
        }

    def evaluate_generation(self, adapter_path, dataset, batch_size=4):
        """Для SAMSum: Perplexity и BLEU"""
        cleanup()
        print("Evaluating Generation (PPL & BLEU)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.load_adapter(adapter_path)
        
        self.tokenizer.padding_side = "right"
        def tokenize_for_ppl(examples):

            full_texts = [
                i + t for i, t in zip(examples["input_text"], examples["target_text"])
            ]

            inputs = self.tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
            
        ppl_dataset = dataset.map(tokenize_for_ppl, batched=True)
        
        trainer = Trainer(model=model, data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False))
        eval_res = trainer.evaluate(ppl_dataset)
        perplexity = math.exp(eval_res['eval_loss'])

        self.tokenizer.padding_side = "left"
        bleu_metric = evaluate.load("sacrebleu")
        
        pipe = pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer,
            device_map="auto", return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        predictions = []
        references = []
        times = []
        
        for out, target in tqdm(zip(pipe(KeyDataset(dataset, "input_text"), batch_size=batch_size, max_new_tokens=50, do_sample=False), subset["target_text"]), total=len(subset)):
            start_t = time.time()
            predictions.append(out[0]['generated_text'].strip())
            references.append([target])
            times.append(time.time() - start_t)

        bleu_score = bleu_metric.compute(predictions=predictions, references=references)

        return {
            "perplexity": perplexity,
            "bleu": bleu_score['score'],
            "inference_latency_ms": (np.mean(times) / batch_size) * 1000
        }