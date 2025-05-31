
# 🧠 PDF Summarizer with LLaMA + llama-cpp-python

This project summarizes large PDF documents using a local LLaMA model (e.g., Mistral or LLaMA2) with recursive chunking and GPU-accelerated inference via `llama-cpp-python`.

---

## 📦 Features

- Summarizes large PDF documents using chunking and recursive summarization.
- Runs local LLaMA models in GGUF format using `llama-cpp-python`.
- GPU acceleration using CUDA/cuBLAS for performance.
- Avoids redundant computation with checkpointing.
- Fully configurable via command-line arguments.

---

## 🔧 Setup Instructions
### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install CUDA 12 (Ubuntu)

```bash
# Prioritize NVIDIA packages
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Fetch NVIDIA keys
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub


# Add NVIDIA repos
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-drivers cuda-12-6 cudnn9-cuda-12-6 libcudnn8-dev libnccl2 libnccl-dev
sudo reboot
```

After reboot:

```bash
nvidia-smi
nvcc --version
```

---

### 3. Install Python Dependencies

#### llama-cpp-python (GPU/cuBLAS):

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --no-cache-dir --force-reinstall llama-cpp-python
```

#### Other dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Download the Mistral-7B-Instruct GGUF model from Hugging Face

Go to [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) and download your preferred GGUF file, for example:

```bash
mkdir -p models/mistral
cd models/mistral
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
cd ../../
```

### Run summarizer with your PDF and model

```bash
python summarize_pdf.py --pdf-path path/to/document.pdf --model-path models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Optional flags:

- `--max-ctx-tokens`: Override context length (default: 6144).
- `--gpu-layers`: Number of layers offloaded to GPU (default: 20).

Example:

```bash
python summarize_pdf.py \
  --pdf-path document.pdf \
  --model-path models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --max-ctx-tokens 4096 \
  --gpu-layers 10
```

---

## 🧠 Notes

- Intermediate summaries are saved in `partial_summaries/`.
- Debug output is saved in `debug_outputs/`.
- Final summary saved to `final_summary.txt`.

---

## 📁 Directory Structure

```
.
├── summarize_pdf.py
├── final_summary.txt
├── partial_summaries/
├── debug_outputs/
└── README.md
```

---

## ✅ Tested On

- Ubuntu 22.04
- Python 3.10
- CUDA 12.9
- NVIDIA RTX 3050 ti

---

## 🛡 License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](LICENSE) file for more details.

> You are free to use, modify, and distribute this software under the terms of the GPLv3. Any derivative work must also be distributed under the same license.

---

## 🙏 Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Mistral models](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
