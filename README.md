# HBQ: Hierarchical scale Block Quantization

## Setup conda environment
```bash
conda env create -f environment.yml
conda activate hbq
```

Build the local CUDA extension for FP16 and psum quantization kernels:

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
python main.py --config_dir config/llama3.1-8b-ins_mmlu.yaml --quant_config config/hbq_a_kv.yaml
```

## Nuances didn't mention in the paper:
- When using FP8-scale under high activation precision (>= 6-bit), FP8-scale face similar scaling factor quantization error as the floor/rounding issue in PoT-scale. So we use *use_ceil* to calculate FP8 scaling when activation is >= 6-bit. See quantizer.py MXFPQuantizer.get_shared_scale() for how the scaling factor is calculated. *use_ceil* is set by quantization configuration yml file with *block_quant/act_scale_use_ceil* and *block_quant/wgt_scale_use_ceil*.
- For PoT-round (mentioned in Section 2.2.1 and Figure 3), we found out that only qk projection benefits from rounding, PoT-floor is better for other layers. So we only use rounding in QK. Rounding is set by quantization configuration yml file with *block_quant/qk_mx_round*.

## Artifact Reproduction

See script/README.md