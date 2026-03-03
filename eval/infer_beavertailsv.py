"""
Inference script for BeaverTails-V dataset evaluation.
Runs model inference on safety-related image-text pairs.
"""

import os
import json
import base64
import requests
import argparse
import math
from tqdm import tqdm
from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path
from typing import Union
import concurrent.futures
from datasets import load_dataset

from utils.prompts import SYSTEM_PROMPT_THINKING
from utils.image_utils import encode_image

# Type alias for image inputs
ImageLike = Union[str, Path, Image.Image, bytes, bytearray, memoryview]


def build_messages(image_path, question):
    """Build chat messages with image and question."""
    image_base64 = encode_image(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_THINKING},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": question}
            ]
        }
    ]


def call_vllm(
    messages,
    model_id,
    temperature=0.0,
    max_tokens=2048,
    api_url="http://localhost:8000/v1/chat/completions",
):
    """Call vLLM API for inference."""
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.1,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=5000)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_exp(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    port=8000,
    output_dir="output",
    max_workers=16,
    max_tokens=2048,
):
    """Run inference experiment on BeaverTails-V dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from local path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "safe_dataset", "data")
    eval_data = load_dataset(dataset_path)["test"]

    api_url = f"http://localhost:{port}/v1/chat/completions"
    outputs = {}

    def run_single_chat(i, d):
        prompt = d["question"]
        image_path = d["image"]
        try:
            output = call_vllm(
                build_messages(image_path, prompt),
                model_id,
                max_tokens=max_tokens,
                api_url=api_url,
            )
            return {
                "id": i,
                "question": prompt,
                "answer": output,
            }
        except Exception as e:
            print({"error": str(e), "question": prompt})
            return {"id": i, "question": prompt, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {
            executor.submit(run_single_chat, i, d): i
            for i, d in enumerate(eval_data)
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_qid), total=len(future_to_qid)
        ):
            qid = future_to_qid[future]
            result = future.result()
            outputs[qid] = result

    # Sort outputs by ID
    outputs = dict(
        sorted(
            outputs.items(),
            key=lambda x: int(str(x[0])) if str(x[0]).isdigit() else str(x[0]),
        )
    )
    
    output_path = os.path.join(output_dir, "beavertailsv_result.json")
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on BeaverTails-V dataset")
    parser.add_argument("--model_name", type=str, default="qwen2_5_vl_3b",
                        help="Model name key (see utils/mllm_map.py)")
    parser.add_argument("--dataset_name", type=str, default="beavertailsv",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--port", type=int, default=8200,
                        help="vLLM server port")
    parser.add_argument("--max_workers", type=int, default=200,
                        help="Number of parallel workers")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Direct model ID (overrides model_name)")
    args = parser.parse_args()

    from utils.mllm_map import mllm_to_module

    if args.model_id:
        model_id = args.model_id
    else:
        model_id = mllm_to_module.get(args.model_name, args.model_name)
    
    output_dir = os.path.join(args.output_dir, args.dataset_name, args.model_name)
    run_exp(
        model_id=model_id,
        port=args.port,
        output_dir=output_dir,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens
    )
