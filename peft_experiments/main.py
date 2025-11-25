import argparse
import os
from data_utils import get_tokenized_dataset
from experiment import ExperimentRunner
from utils import save_results

def main():
    parser = argparse.ArgumentParser(description="Run PEFT experiments")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="HF Model ID")
    parser.add_argument("--task", type=str, choices=["mrpc", "samsum", "all"], default="all", help="Task to run")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--methods", nargs="+", default=["lora", "qlora", "dora"], help="Methods to test")
    parser.add_argument("--trained", type=bool, default=False, help="If you have trained models")
    args = parser.parse_args()

    runner = ExperimentRunner(args.model_id)
    
    # --- Experiment 1: MRPC (Classification) ---
    if args.task in ["mrpc", "all"]:
        print("\n\n========== RUNNING MRPC EXPERIMENT ==========")
        raw_ds, tokenized_ds = get_tokenized_dataset("SetFit/mrpc", runner.tokenizer)
        
        train_ds = tokenized_ds["train"]
        val_ds_tok = tokenized_ds["validation"]
        test_ds_tok = tokenized_ds["test"]

        if args.trained:
            for path in ['output/lora/final_adapter', 'output/qlora/final_adapter', 'output/dora/final_adapter']:
                
                metrics = runner.evaluate_classification(path, test_ds_tok)
                
                final_report = {
                    "dataset": "MRPC",
                    "model": args.model_id,
                    **metrics
                }
        else:
            for method in args.methods:
                res = runner.train(method, train_ds, val_ds_tok, args.batch_size, args.epochs)
                
                metrics = runner.evaluate_classification(res["adapter_path"], test_ds_tok)
                
                final_report = {
                    "dataset": "MRPC",
                    "model": args.model_id,
                    **res,
                    **metrics
                }
                save_results(final_report)

    # --- Experiment 2: SAMSum (Summarization) ---
    if args.task in ["samsum", "all"]:
        print("\n\n========== RUNNING SAMSUM EXPERIMENT ==========")
        raw_ds, tokenized_ds = get_tokenized_dataset("knkarthick/samsum", runner.tokenizer)
        
        train_ds = tokenized_ds["train"].select(range(2000))
        val_ds_tok = tokenized_ds["validation"]
        test_ds_tok = tokenized_ds["test"]

        for method in args.methods:
            res = runner.train(method, train_ds, val_ds_tok, args.batch_size, args.epochs)
            
            metrics = runner.evaluate_generation(res["adapter_path"], test_ds_tok)
            
            final_report = {
                "dataset": "SAMSum",
                "model": args.model_id,
                **res,
                **metrics
            }
            save_results(final_report)

if __name__ == "__main__":
    main()