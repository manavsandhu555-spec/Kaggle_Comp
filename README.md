# Facial Feature Classification — setup and run (VS Code, Windows, NVIDIA GPU)

## Files
| File | What it is |
|---|---|
| `facial_feature_classification.ipynb` | The whole pipeline, organised by assessment Task 1–6 |
| `ffc_lib.py` | Dataset, transforms, models, training / evaluation loops, Grad-CAM (imported by the notebook) |
| `requirements.txt` | Python packages (PyTorch is installed separately, see below) |

Keep all three in the **kaggle comp** folder, next to the extracted dataset folder.

## 1. One-time setup
1. Install **Python 3.11 or 3.12** (python.org) and, in VS Code, the **Python** and **Jupyter** extensions.
2. Make sure the NVIDIA driver is up to date (`nvidia-smi` in a terminal should show your GPU).
3. In VS Code: *File → Open Folder… → kaggle comp*, then open a terminal (Ctrl + `) and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
# PyTorch with CUDA: copy the exact command from https://pytorch.org/get-started/locally
# (Stable, Windows, Pip, Python, CUDA). It looks like:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and your GPU name. If it prints `False`, you installed the CPU-only build — uninstall torch/torchvision and reinstall with the CUDA command.

If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 2. Run the notebook
1. Extract the Kaggle zip inside **kaggle comp** (right-click → *Extract All*). `DATA_DIR` in the config cell points at that folder; the code finds `train/`, `test/` and the sample submission inside it automatically.
2. Open `facial_feature_classification.ipynb`, click **Select Kernel** (top right) → *Python Environments* → `.venv`.
3. **Smoke test first:** in the config cell set `QUICK_RUN = True`, then *Run All*. It trains on a tiny subset for 2 epochs (writes to `outputs_quick/`) just to prove everything works.
4. **Real run:** set `QUICK_RUN = False` and *Run All*. Training checkpoints, learning-curve CSVs, metrics, `submission.csv` and all report figures are written to `outputs/` (figures in `outputs/figures/`).
   * Want only the main model? Set `RUN_ABLATIONS = False` (trains E4 only).
   * Already-trained experiments are re-loaded from their checkpoints, so re-running is fast. Set `RETRAIN = True` to train again.
5. Upload `outputs/submission.csv` to Kaggle (*Submit Prediction*), screenshot the result, and type your score into `KAGGLE_PUBLIC_SCORE` in Section 5.4, then re-run the cells below it.

## Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA out of memory` | lower `BATCH_SIZE` to 16 (or `IMG_SIZE` to 192) in the config cell |
| DataLoader hangs / `BrokenPipeError` / pickling errors | set `NUM_WORKERS = 0` |
| `IProgress not found` | `pip install ipywidgets`, then restart the kernel |
| Pretrained weights fail to download | needs internet the first time (timm pulls weights from Hugging Face) |
| `Could not find labelled training data` | check `DATA_DIR` points at the extracted folder |

## Reproducibility
Fixed seed (42) for Python / NumPy / PyTorch / DataLoader workers, deterministic cuDNN, saved train/val split (`outputs/train_split.csv`, `val_split.csv`), saved label map, per-epoch histories and JSON summaries for every experiment. GPU kernels can still give tiny run-to-run differences (±0.1–0.3 % accuracy).
