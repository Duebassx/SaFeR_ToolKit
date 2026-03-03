# SaFeR-ToolKit

**Structured Reasoning via Virtual Tool Calling for Multimodal Safety**

## Features
- **RL Training**: GRPO-based safety alignment training
- **Evaluation**: Safety evaluation scripts for multimodal benchmarks

## Requirements

- Python 3.9+
- CUDA 12.x
- PyTorch 2.x
- vLLM >= 0.8.0
- transformers >= 4.54.0

## Installation

### Option 1: Local Installation

```bash
cd SaFeR-ToolKit
pip install -e .
```

### Option 2: Docker (Recommended)

This project is built upon [EasyR1](https://github.com/hiyouga/EasyR1), which provides official Docker images for a consistent and hassle-free environment setup.

## Quick Start

### 1. RL Training

Configure your training data and run:

```bash
# Set your data paths
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_DATA="path/to/train.json"
VAL_DATA="path/to/val.json"

# Run training
bash examples/safer_toolkit.sh
```

#### Training Data Format

Your training data should be in JSON format:
```json
[
  {
    "id": "sample_001",
    "images": ["path/to/image.jpg"],
    "problem": "<image>\nYour question here",
    "answer": ""
  }
]
```

#### Merge Model Weights

After training, merge the distributed model shards into a single HuggingFace model:

```bash
python scripts/model_merger.py --local_dir <checkpoint_path>
```

The merged model will be saved to `<checkpoint_path>/huggingface/`.

### 2. Evaluation

#### Step 1: Start vLLM Server

Use the provided script to start the vLLM server:

```bash
bash scripts/lauch_server.sh
```

#### Step 2: Run Inference

```bash
cd eval

python infer_beavertailsv.py \
    --model_name qwen2_5_vl_3b \
    --port 8200 \
    --max_workers 100 \
    --output_dir output
```

#### Step 3: Run Evaluation

Set environment variables for the judge model:
```bash
export BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
```

Run evaluation:
```bash
python eval_beavertailsv.py \
    --model_name qwen2_5_vl_3b \
    --output_dir output \
    --outeval_dir eval_out \
    --judge_model gpt-5-mini \
    --max_workers 50
```

## Project Structure

```
SafeR-ToolKit/
├── verl/                          # Core RL training framework
│   ├── trainer/                   # Training logic
│   ├── workers/                   # Actor, Critic, Reward workers
│   └── utils/                     # Utilities
├── examples/
│   ├── safer_toolkit.sh           # Training script
│   ├── reward_function/           # Custom reward functions
│   │   └── safer_toolkit.py       # Safety-aware reward
│   └── format_prompt/             # Prompt templates
│       └── toolkit.jinja          # Default prompt template
├── eval/
│   ├── infer_beavertailsv.py      # Inference script
│   ├── eval_beavertailsv.py       # Evaluation script
│   └── utils/                     # Evaluation utilities
├── scripts/
│   ├── lauch_server.sh            # vLLM server startup script
│   └── model_merger.py            # Merge distributed model weights
├── requirements.txt               # Dependencies
└── setup.py                       # Installation
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This project is built upon [EasyR1](https://github.com/hiyouga/EasyR1). We thank the authors for their excellent work.
