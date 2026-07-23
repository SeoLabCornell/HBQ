# RTL Implementation of all PE modules

## 📂 Repository Overview

This repository provides the hardware artifacts for **"HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference"**. It includes baseline Processing Element (PE) designs for BQ baselines, including all designs explored in *Section 4. BQ Design Space Exploration* such as MXFP4 and NVFP4 as well as Amove, and also HBQ PE design.  
---

## 🏛️ Directory Structure

## 1. Baseline Block Quantization Processing Elements (BQ PE)

![Baseline PE Design](PE_baseline.png)

Baseline BQ PE. We have three PE designs to cover different exploration space. *params.vh* is the configuration file for each PE to tune different knobs.

In the main module SystemVerilog file MAC*.sv, MAC module with *_MX postfix means it uses PoT-scale, *_NV postfix means it uses FP8-scale, without any of two means it can be reconfigurated by the argument. You can simply run syn_FP8.tcl or syn_PoT.tcl to change the scaling scheme.
---

### 1.1 BQ/MAC_WXAY

Basic BQ PE module that performs BQ MAC with **FP format with E2MX**. FP with 2-bit exponent is the optimal configuration we found in most of the BQ setting (see paper Section 4.2). For instance if W4A8 is set, then the module uses FP4 w/ E2M1 on weight and FP8 w/ E2M5 on activation. This module allows user to tune following knobs (variable names in params.vh shown in parenthesis):
1. Weight precision (parameter int X)
2. Activation precision (parameter int Y)
3. Block size (parameter int B)
4. Scaling methods: PoT-scale or FP8-scale (tune this knob by synthesize different MAC, _MX and _NY variation)

Paper result reproduction: This module covers Fig3, Fig 4(b), Figure 5, Figure 6, and Table 9 (area for MXFP/NVFP).

### 1.2 BQ/MAC_EM
Basic BQ PE module using FP format that allows user to tune different exponent and mantissa combination. Tunable knobs (variable names in params.vh shown in parenthesis):
1. Activation precision(integer int Y)
2. Activation exponent bit (integer int E), mantissa bit is adjusted by Y and E
3. Block size (parameter int B)
4. Scaling methods: PoT-scale or FP8-scale (tune this knob by synthesize different MAC, _MX and _NY variation)

The weight format is fixed to FP4 E2M1.

Paper result reproduction: This module covers Fig 4(b), Figure 5.

### 1.3 BQ/MAC_int
Basic BQ PE module using integer for both activation and weight. Tunable knobs:
1. Activation precision 
2. Weight precision
3. Block size
4. Scaling methods (PoT-scale or FP8-scale)

Paper result reproduction: This module covers Fig 4(b), Figure 5.

## 2. HBQ PE
Implementation of HBQ PE. *params.vh* is the configuration file for each PE to tune different knobs.

You can tune *SUB_B* in params.vh to reproduce HBQ-E and HBQ-A.


## Paper Result Reproduction Guideline
To run the synthesis, please setup proper path to your SDK in .tcl and run:
` bash
dc_shell -f .syn_FP8.tcl # for FP8-scale
dc_shell -f .syn_PoT.tcl # for PoT-scale
`
Please adjust param.vh for different tuning knobs.

### Figure 3
1. Modify compile with w4a4_params.vh 

### 1.1. `Baseline/Amove/`  
#### **W4A4B4 INT PE (NV scheme)**

Implements the *Amove* architecture using **4-bit weights**, **4-bit activations**, and **block size B = 4** under the NV quantization scheme.
#### Reference
> Xie, Xilong, et al. "Amove: Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type." *Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture®*. 2025.

#### Key Modules
- **`MAC_Amove.sv`** – INT MAC supporting NV mode for Amove architecture.  
- **`MUL_int.sv`** – INT Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling.  
- **`FP_ACCUM.v`** – FP partial sum accumulator with external partial sum buffer. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Configuration for INT NV W4A4B4 operation.

---

### 1.2. `Baseline/MAC_WXAY_MX/`
#### **WXAY FP PE (PoT-scale)**

A floating-point PE supporting configurable FP formats of the form **WXAY** (X: weight bits, Y: activation bits).

Used configurations in evaluations:
- **W4A4B32** for MXFP energy breakdown 

#### Key Modules
- **`MAC_WXAY_MX.sv`** – FP MAC supporting MX mode.  
- **`MUL_WXAY.sv`** – FP Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling. 
- **`FP_ACCUM.v`** – FP partial sum accumulator with external partial sum buffer. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Parameter file defining X, Y, B.

---

### 1.3. `Baseline/MAC_WXAY_NV/`
#### **WXAY FP PE (NV scheme)**

A floating-point PE supporting configurable FP formats of the form **WXAY** (X: weight bits, Y: activation bits).

Used configurations in evaluations:
- **W4A4B16** for NVFP energy breakdown 

#### Key Modules
- **`MAC_WXAY_NV.sv`** – FP MAC supporting NV mode.  
- **`MUL_WXAY.sv`** – FP Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling. 
- **`FP_ACCUM.v`** – FP partial sum accumulator with external partial sum buffer. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Parameter file defining X, Y, B.

---

## 2. Design Space Exploration

The `DSE/` directory contains multiple versions of compute engines designed for precision sweeping.  
All parameters (bit widths, exponent/mantissa widths, block sizes) are defined in `params.vh`.

---

### 2.1. `DSE/MAC_EM/`  
#### **FP8 PE (W4A8) with E/M sweep — MX & NV**

Floating-point PE supporting a family of FP8 formats via configurable exponent (E) and mantissa (M) lengths.

#### Sweepable Parameters
- **Exponent bits (E):** {2, 3, 4, 5}  
- **Mantissa bits (M):** {5, 4, 3, 2}

#### Key Modules
- **`MAC_EM.sv`** – FP MAC supporting MX and NV mode, with configurable E, M modification.  
- **`MUL_EM.sv`** – FP Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling. 
- **`FP_ACCUM.v`** – FP partial sum accumulator for. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Parameter file defining X, Y, B, **E, M**. (X, Y) is fixed to (4, 8).


---

### 2.2. `DSE/MAC_int/`  
#### **WXAY Integer PE — MX & NV**

Configurable integer WXAY MAC for sweeping integer precision, block size, and quantization format.

#### Sweepable Parameters
- **Weight precision (X bits)**  
- **Activation precision (Y bits)**  
- **Block size (B)**  

#### Key Modules
- **`MAC_int.sv`** – INT MAC supporting MX and NV mode.  
- **`MUL_int.sv`** – INT Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling. 
- **`FP_ACCUM.v`** – FP partial sum accumulator. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Parameter file defining X, Y, B.

---

### 2.3. `DSE/MAC_WXAY/`  
#### **WXAY Floating-Point PE — MX & NV**

A flexible floating-point MAC supporting block-shared exponent MX/NV quantization with different bit width of weight and activation.

#### Sweepable Parameters
- **Weight FP width (X bits)**  
- **Activation FP width (Y bits)**  
- **Block size (B)**  

#### Key Modules
- **`MAC_WXAY.sv`** – FP MAC supporting MX and NV mode.  
- **`MUL_WXAY.sv`** – FP Multiplier and Adder Tree for block accumulation.  
- **`DEQUANT.v`** – Dequantization logic restoring intra-block weight scaling. 
- **`FP_ACCUM.v`** – FP partial sum accumulator. Excluded subnormal expression for HW efficiency.
- **`params.vh`** – Parameter file defining X, Y, B.

### Dataflow and Register Amortization

---