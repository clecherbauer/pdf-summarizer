import os
import shutil
import textwrap
import argparse
from pypdf import PdfReader
from llama_cpp import Llama

# optimized for RTX 5030 Ti
DEFAULT_MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_MAX_CTX_TOKENS = 6144
DEFAULT_GPU_LAYERS = 20
PARTIAL_DIR = "partial_summaries"
DEBUG_DIR = "debug_outputs"

def save_debug_text(filename, text):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    save_debug_text("original_text.txt", text)
    return text

def chunk_text(text, max_chars=1500):
    chunks = textwrap.wrap(text, width=max_chars)
    for i, chunk in enumerate(chunks, 1):
        save_debug_text(f"chunk_{i}.txt", chunk)
    return chunks

def load_model(model_path, max_ctx_tokens, gpu_layers):
    return Llama(model_path=model_path, n_ctx=max_ctx_tokens, n_gpu_layers=gpu_layers)

def summarize(text, llm):
    prompt = f"Summarize the following text in bullet points:\n\n{text.strip()}\n\nSummary:"
    response = llm(prompt, max_tokens=512)
    summary = response["choices"][0]["text"].strip()
    return summary

def recursive_summarize(texts, llm, max_chunk_chars=1500, level=1):
    combined_text = "\n\n".join(texts)
    save_debug_text(f"level_{level}_combined_input.txt", combined_text)

    if len(combined_text) <= max_chunk_chars:
        summary = summarize(combined_text, llm)
        save_debug_text(f"level_{level}_summary.txt", summary)
        return summary
    else:
        chunks = chunk_text(combined_text, max_chars=max_chunk_chars)
        partial_summaries = []
        for i, chunk in enumerate(chunks, 1):
            summary = summarize(chunk, llm)
            save_debug_text(f"level_{level}_chunk_{i}_summary.txt", summary)
            partial_summaries.append(summary)
        return recursive_summarize(partial_summaries, llm, max_chunk_chars, level + 1)

def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF document using a local LLaMA model.")
    parser.add_argument("--pdf-path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the LLaMA model GGUF file.")
    parser.add_argument("--max-ctx-tokens", type=int, default=DEFAULT_MAX_CTX_TOKENS, help="Max context tokens for the model.")
    parser.add_argument("--gpu-layers", type=int, default=DEFAULT_GPU_LAYERS, help="Number of model layers to offload to GPU.")
    args = parser.parse_args()

    max_chars_per_chunk = args.max_ctx_tokens * 4

    if os.path.exists(PARTIAL_DIR):
        answer = input(f"Partial summaries exist in '{PARTIAL_DIR}'. Restart and overwrite partials? (y/N): ").strip().lower()
        if answer == "y":
            print("Deleting existing partial summaries...")
            shutil.rmtree(PARTIAL_DIR)
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    print("Loading PDF...")
    original_text = extract_text_from_pdf(args.pdf_path)

    print("Splitting text into chunks...")
    chunks = chunk_text(original_text, max_chars=max_chars_per_chunk)

    print(f"Summarizing {len(chunks)} chunks...")
    llm = load_model(args.model_path, args.max_ctx_tokens, args.gpu_layers)

    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        part_file = os.path.join(PARTIAL_DIR, f"summary_part_{i}.txt")
        if os.path.exists(part_file):
            print(f"✔️ Chunk {i} already summarized. Skipping...")
            with open(part_file, "r", encoding="utf-8") as f:
                partial_summaries.append(f.read())
        else:
            print(f"🧠 Summarizing chunk {i}...")
            summary = summarize(chunk, llm)
            partial_summaries.append(summary)
            with open(part_file, "w", encoding="utf-8") as f:
                f.write(summary)

    print("Creating recursive final summary...")
    final_summary = recursive_summarize(partial_summaries, llm, max_chunk_chars=max_chars_per_chunk)

    final_path = "final_summary.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"\n✅ Final summary saved to '{final_path}'\n")

    print(final_summary)

if __name__ == "__main__":
    main()
