# HBQ: Hierarchical scale Block Quantization

## Setup
```bash
conda env create -f environment.yml
conda activate hbq
```

Build the local CUDA extension:

```bash
cd blockquant_ext_src
pip install ninja setuptools wheel
pip install -e . --no-build-isolation
cd ..
```

## Usage
```bash
python main.py --config_dir {model/benchmark config yaml} --quant_config {quantization config yaml}

# Example: Llama3-8B with HBQ-A on wikitext
python main.py --config_dir config/llama3-8b_wiki.yaml --quant_config config/hbq_a.yaml

# Llama3.1-8B-instruct with HBQ-E w/ KV cache quantization on MMLU
python main.py --config_dir config/llama3.1-8b-inst_mmlu.yaml --quant_config config/hbq_a_kv.yaml
```
