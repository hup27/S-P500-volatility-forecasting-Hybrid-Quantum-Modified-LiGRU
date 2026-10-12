import os, time, psutil, torch
from xml.parsers.expat import model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_mb():
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return ram_mb, gpu_mb

def benchmark_latency(model, sample, device, repeats=100):
    model.eval()
    sample = sample.to(device)

    with torch.no_grad():
        for _ in range(10):
            h = torch.zeros(model.num_layers, sample.size(0), model.hidden_dim, device=sample.device)
            _ = model(sample, h)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            h = torch.zeros(model.num_layers, sample.size(0), model.hidden_dim, device=sample.device)
            _ = model(sample, h)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed / repeats * 1000