

# Supermix_29

Supermix_29 is a hybrid local-AI repository that combines:

1. **Supermix / ChampionNet research and inference assets** for expert-routed chat and retrieval experiments.
2. **Qwen-based local desktop packaging** for a bundled Windows chat application that launches a local web UI/server.
3. **Static browser deployment assets** for GitHub Pages-compatible metadata-driven chat.

The repo is not just a model drop. It contains training code, inference runtimes, dataset builders, packaging scripts, documentation, browser deployment assets, generated state/output folders, and Windows installer files.

---

## What this repository contains

### 1) Supermix / ChampionNet model work
The internal architecture docs describe a **12-layer ChampionNet backbone** with a swappable expert-routed classifier layer. The documented model family evolves from simple classifier heads into Mixture-of-Experts variants such as noisy top-k gating, hierarchical routing, expert-choice routing, sigma gating, and recursive thought-style routing.

### 2) Runtime inference bundle
The `runtime_python/` folder contains a packaged local runtime with:
- model weights
- metadata JSON
- terminal and web launchers
- runtime requirements
- inference-critical Python modules

This is the fastest place to start if you want to run the project locally.

### 3) Qwen desktop application packaging
The repo also contains `qwen_chat_desktop_app.py`, `qwen_chat_web_app.py`, desktop build scripts, installer scripts, and Inno Setup definitions for producing a Windows desktop app that starts a local chat server automatically.

### 4) Static GitHub Pages chat
The `web_static/` folder contains a browser-only metadata-driven chat UI for GitHub Pages. It does **not** execute `.pth` PyTorch weights in-browser; instead it uses browser-readable metadata JSON.

---

## Naming note

The repository is named **Supermix_29**, but several internal documents and model files still refer to:
- **Supermix_27**
- **v28 UltraExpert**
- **ChampionNet**
- **Supermix Qwen Desktop**

That is reflected in the current repo contents. This README treats **Supermix_29** as the repository name and **Supermix / ChampionNet / v28** as the model lineage inside the repo.

---

## Repository layout

```text
Supermix_29/
├── source/                     # Main research/training/dev workspace
│   ├── run.py                 # ChampionNet backbone definitions
│   ├── model_variants.py      # Expert head variants and architectural variants
│   ├── finetune_chat.py       # Fine-tuning / preference-optimized training entrypoint
│   ├── chat_pipeline.py       # Feature building, labels, dataset pipeline
│   ├── qwen_chat_web_app.py   # Qwen-based local web chat app
│   ├── qwen_chat_desktop_app.py
│   ├── qwen_supermix_pipeline.py
│   ├── build_*                # Dataset builders + desktop build scripts
│   ├── requirements_train_build.txt
│   └── launch_*               # Batch/PowerShell helpers
│
├── runtime_python/            # Packaged local inference runtime
│   ├── champion_model_chat_supermix_v27_500k_ft.pth
│   ├── chat_model_meta_supermix_v27_500k.json
│   ├── chat_app.py
│   ├── chat_pipeline.py
│   ├── chat_web_app.py
│   ├── launch_chat_terminal_supermix.bat
│   ├── launch_chat_web_supermix.bat
│   └── requirements_runtime_interface.txt
│
├── web_static/                # Browser-only GitHub Pages deployment
│   ├── index.html
│   ├── chat_model_meta_supermix_v27_500k.browser.json
│   └── README_GITHUB_PAGES.txt
│
├── installer/                 # Windows installer assets
│   ├── SupermixQwenDesktop.iss
│   └── postinstall_notes.txt
│
├── artifacts/                 # Training outputs / adapters / checkpoints
├── datasets/                  # Dataset storage
├── research/                  # Research-side notes/assets
├── dist/                      # Packaged desktop outputs
├── build/                     # Build staging/output
├── ARCHITECTURE.md            # Technical architecture guide
├── MODEL_CARD_V28.md          # Model card
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
└── LICENSE
````

---

## Core capabilities

* Local Python chat runtime
* Expert-routed model experimentation
* Fine-tuning and dataset-building workflows
* Static browser chat for GitHub Pages
* Windows desktop EXE packaging
* Windows installer generation with Inno Setup
* Cross-device runtime resolution with preference order across CUDA / NPU / XPU / DirectML / MPS / CPU

---

## Quick start

## Run the packaged local runtime

Install runtime dependencies:

```bash
python -m pip install -r runtime_python/requirements_runtime_interface.txt
```

Start the local web runtime:

```bash
python runtime_python/chat_web_app.py
```

Or use the bundled launchers on Windows:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

Expected behavior:

* loads the packaged runtime checkpoint and metadata
* starts a local server on `http://127.0.0.1:8000` or `http://localhost:8000`
* serves a local chat UI for inference/testing

---

## Run the static browser version

The static version is for **metadata-driven browser retrieval**, not full PyTorch inference.

Files:

* `web_static/index.html`
* `web_static/chat_model_meta_supermix_v27_500k.browser.json`

To use it locally, open `index.html` in a browser and load the metadata JSON.

To publish on GitHub Pages:

1. Upload `index.html` and `chat_model_meta_supermix_v27_500k.browser.json`.
2. Enable GitHub Pages for that branch/folder.
3. Open the published site.
4. Load metadata in the UI.

### Important limitation

GitHub Pages / browser JavaScript does **not** execute the `.pth` model directly. The static version uses metadata JSON instead.

---

## Build the Windows desktop EXE

The desktop build system packages a local webview app and bundles the latest adapter artifact.

### Prerequisites

* Python on `PATH`
* Windows
* dependencies used by the build script:

  * `pywebview`
  * `pillow`
  * `pyinstaller`

### Build command

```powershell
powershell -ExecutionPolicy Bypass -File build_qwen_chat_desktop_exe.ps1
```

Output:

```text
dist\SupermixQwenDesktop\SupermixQwenDesktop.exe
```

### What the EXE build does

* generates desktop branding assets
* finds the latest adapter under the project artifacts
* stages a desktop bundle
* packages the local app with PyInstaller
* embeds the chat UI/webview launcher
* includes assets and bundled artifact metadata

---

## Build the Windows installer

The repo includes an Inno Setup script under `installer/SupermixQwenDesktop.iss`.

### Prerequisites

* Build the desktop EXE first, or let the installer script do it
* Install **Inno Setup 6**

Install Inno Setup with:

```powershell
winget install --id JRSoftware.InnoSetup -e --accept-package-agreements --accept-source-agreements
```

### Build command

```powershell
powershell -ExecutionPolicy Bypass -File build_qwen_chat_desktop_installer.ps1
```

Installer output is written under:

```text
dist\installer\
```

---

## Desktop installer runtime notes

The post-install notes indicate that the desktop package:

* installs a lightweight launcher executable
* expects Python to be installed and available as `python`
* bundles the adapter
* starts the local chat server automatically when opened
* expects a local `Qwen2.5-0.5B-Instruct` base snapshot to exist in the Hugging Face cache unless you change the base-model path in your workflow

---

## Training and fine-tuning

Install training/build dependencies:

```bash
python -m pip install -r source/requirements_train_build.txt
```

Current training requirements are lightweight at repo level:

* `torch`
* `pillow`
* `nltk`

### Example fine-tune command

```bash
cd source
python finetune_chat.py --data conversation_data.jsonl --weights ../runtime_python/champion_model_chat_supermix_v27_500k_ft.pth --model_size smarter_expert --epochs 8 --batch_size 32
```

### Training pipeline features

The architecture docs describe a training stack with:

* JSONL dataset ingestion
* label assignment
* feature-mode selection
* stratified splitting
* optional weighted sampling
* cross-entropy plus preference optimization
* AdamW
* cosine LR scheduling
* EMA
* gradient clipping
* AMP on CUDA devices

### Feature modes

Documented feature modes include:

* `legacy`
* `context_v2`
* `context_v3`
* `context_v4`
* `context_v5`
* `context_mix_v1`
* `context_mix_v2_mm`

---

## Architecture overview

The documented ChampionNet architecture uses:

* **12 total sequential layers**
* **layers 0–9**: gated feed-forward feature extraction
* **layer 10**: swappable classifier / expert head
* **layer 11**: final output normalization/projection
* **256-dimensional input features**
* **10 output classes**

The project documents several routing strategies and architectural ideas:

* noisy top-k gating
* hierarchical MoE routing
* auxiliary-loss-free dynamic expert bias balancing
* expert-choice routing
* sigma gating
* recursive multi-step expert reasoning

---

## Model / head variants

The documented Supermix family includes:

### `ultra_expert`

Primary v28-style noisy top-k expert head with heterogeneous experts and residual calibration.

### `hierarchical_expert`

Two-level routing with domain groups, shared always-on expert, and per-expert LoRA-style correction.

### `deep_expert`

Shared expert + routed experts with dynamic bias updates for load balancing without relying on an auxiliary balancing loss.

### `expert_choice`

Experts select tokens rather than tokens selecting experts.

### `smarter_expert`

Independent sigmoid-gated multi-expert activation with LoRA adapters.

### `thought_expert`

Iterative reasoning loop with cross-expert attention fusion.

### `recursive_expert`

Recursive reasoning with adaptive early exit, multi-head routing, and hierarchical shared experts.

---

## Device resolution

The runtime documentation describes this device resolution order:

```text
cuda -> npu -> xpu -> dml -> mps -> cpu
```

That makes the repo relevant for local Windows experimentation across:

* NVIDIA CUDA
* NPUs where supported
* Intel/XPU-style backends where supported
* DirectML
* Apple MPS
* CPU fallback

---

## Important limitations

* The static browser build is **not** full neural inference.
* The repo contains mixed naming across versions and product surfaces.
* Some workflows are Windows-first, especially desktop and installer packaging.
* Desktop packaging and runtime notes imply local Python and local model/base assets are still part of the expected environment.
* Build, dist, runtime, artifact, and research outputs are all present in-repo, so this is a combined dev/runtime workspace rather than a minimal source-only package.

---

## Recommended docs to read next

* [`ARCHITECTURE.md`](ARCHITECTURE.md) — detailed architectural notes
* [`MODEL_CARD_V28.md`](MODEL_CARD_V28.md) — model card and documented variant behavior
* [`CONTRIBUTING.md`](CONTRIBUTING.md)
* [`SECURITY.md`](SECURITY.md)
* [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

---

## Suggested audience

This repository is best suited for:

* local AI tinkerers
* Windows desktop packaging experiments
* expert-routing / MoE architecture exploration
* chat inference pipeline experimentation
* GitHub Pages metadata-chat deployment
* hybrid research + deployment workflows

---

## License

See [`LICENSE`](LICENSE).

---

## Summary

Supermix_29 is a combined **research, runtime, packaging, and deployment** repository centered on the Supermix / ChampionNet lineage, with additional Qwen-based desktop packaging and a metadata-only browser deployment path. If you want the fastest start, begin with `runtime_python/`. If you want architecture and training work, use `source/`. If you want browser-only hosting, use `web_static/`. If you want a Windows desktop app, use the desktop build and installer scripts.

```

One thing I’d change next is tightening the README around whichever direction you want to emphasize most: **research model repo**, **desktop app repo**, or **portfolio/showcase repo**.
::contentReference[oaicite:1]{index=1}
```

[1]: https://github.com/kai9987kai/Supermix_29 "GitHub - kai9987kai/Supermix_29: Supermix_27 v28 represents a significant architectural evolution of the ChampionNet series, introducing a Mixture-of-Experts (MoE) classifier head for enhanced retrieval and reasoning capabilities. The project is a structured AI/chat repository designed for high-performance inference, featuring a clear split between source code, runtime assets, and · GitHub"
