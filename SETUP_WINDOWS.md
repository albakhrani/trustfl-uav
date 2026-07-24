# Environment Setup — Windows 11, RTX 4080 Laptop GPU

Target machine: Alienware m16 R1, i9-13900HX, 64 GB RAM, RTX 4080 Laptop (12 GB VRAM), Windows 11 Home 25H2.

## 1. Install Python 3.12

Download **Python 3.12.x (64-bit)** from https://www.python.org/downloads/windows/
- During install: check **"Add python.exe to PATH"**, then "Install Now".
- Verify in a new PowerShell window:
  ```powershell
  python --version    # Python 3.12.x
  ```
(Python 3.11 also works. Avoid 3.13+ until PyTorch support stabilizes.)

## 2. NVIDIA driver

You do NOT need to install the CUDA Toolkit — PyTorch wheels bundle their own CUDA runtime. You only need a recent NVIDIA driver:
- GeForce driver ≥ 550 (Game Ready or Studio). Check with:
  ```powershell
  nvidia-smi
  ```
  If this prints a table with your RTX 4080 and a driver version, you're set. If not, install from https://www.nvidia.com/drivers.

## 3. Create the project environment

In PowerShell, from the folder where you unpacked `trustfl-uav/`:
```powershell
cd trustfl-uav
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If PowerShell blocks activation, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Install PyTorch with CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
(cu126 = CUDA 12.6 wheels, correct for RTX 40-series on current drivers. If
this index ever 404s, check https://pytorch.org/get-started/locally/ for the
current CUDA wheel tag and substitute it.)

Then the rest:
```powershell
pip install -r requirements.txt
```

## 5. Verify GPU

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `... True NVIDIA GeForce RTX 4080 Laptop GPU`.

## 6. First real run (MNIST FedAvg sanity check)

```powershell
python -m trustfl.experiments.run --config configs/mnist_fedavg_debug.yaml
```
- First run downloads MNIST (~60 MB) into `./data`.
- Expected: test accuracy **> 0.97 within 50 rounds**, a few minutes on GPU.
- If this passes, the FL core is verified end-to-end on your hardware.

Then the trust framework under attack (3 seeds, ~10–15 min):
```powershell
python -m trustfl.experiments.run --config configs/mnist_trustfl_debug.yaml
```

## Notes

- **VRAM**: MNIST/FEMNIST CNNs use <1 GB. ResNet-18 + CIFAR-100 with batch 64
  uses ~2–3 GB — comfortable within 12 GB. Clients train sequentially (they
  share the GPU), which is the standard FL-simulation pattern.
- **CPU dataloading**: if GPU sits idle, we can add `num_workers` later;
  leave defaults for now (determinism is easier to reason about).
- **Results** land in `./results/<experiment>/<seedN>/` — `metrics.jsonl`
  (per-round), `config_resolved.yaml`, `run_meta.json` (hardware/software/
  seed record for the reproducibility checklist).
