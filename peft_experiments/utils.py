import torch
import gc
import json
import os
from datetime import datetime

class VRAMTracker:
    """Контекстный менеджер для отслеживания пикового потребления VRAM."""
    def __init__(self):
        self.peak_memory = 0

    def __enter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

def cleanup():
    """Очистка памяти GPU."""
    gc.collect()
    torch.cuda.empty_cache()

def save_results(results, filename="experiment_results.json"):
    """Сохраняет результаты в JSON файл."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append(results)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {filename}")