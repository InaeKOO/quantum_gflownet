import torch, time

def fidelity(U, V):
    # assumes both on same device
    return torch.abs((U.conj().transpose(-2,-1) @ V).trace()) / U.shape[-1]

def bench(n, device):
    d = 2**n
    U = torch.randn(d, d, dtype=torch.complex128, device=device)
    V = torch.randn(d, d, dtype=torch.complex128, device=device)
    # warm‑up
    for _ in range(10):
        _ = fidelity(U, V)
    # timed
    torch.cuda.synchronize() if device!="cpu" else None
    t0 = time.time()
    for _ in range(50):
        fid = fidelity(U, V)
    if device!="cpu": torch.cuda.synchronize()
    return (time.time() - t0) / 50

for n in [6, 8, 10, 12]:
    t_cpu = bench(n, "cpu")
    t_gpu = bench(n, "cuda")
    print(f"n={n}, cpu={t_cpu*1e3:.2f} ms, gpu={t_gpu*1e3:.2f} ms")
