import numpy as np
import torch
import random

def torch_init_backend_and_stats():
    torch.jit.enable_onednn_fusion(True)
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

def torch_seed_worker(worker_id):
    #worker_seed = torch.initial_seed() % 2**32
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)

torch_generator = torch.Generator()
torch_generator.manual_seed(42)