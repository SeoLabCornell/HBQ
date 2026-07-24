"""Analytical performance / energy model for the accelerator comparison figures.

Replaces the spreadsheet flow: every number used by script/plot_fig16.py
(system energy breakdown) and script/plot_fig18.py (iso-area speedup) is
computed here from the model architectures and the hardware parameters.

Run directly to print all intermediate numbers:
    python src/perf_model.py
"""

# --------------------------------------------------------------------------- #
# Model architectures                                                          #
# --------------------------------------------------------------------------- #
MODELS = {
    "Llama2-7B":   dict(embed=4096, head_dim=128, heads=32, kv_heads=32, mlp=11008, layers=32, vocab=32000),
    "Llama3-8B":   dict(embed=4096, head_dim=128, heads=32, kv_heads=8,  mlp=14336, layers=32, vocab=128256),
    "Llama3.2-3B": dict(embed=3072, head_dim=128, heads=24, kv_heads=8,  mlp=8192,  layers=28, vocab=128256),
    "Qwen2.5-3B":  dict(embed=2048, head_dim=128, heads=16, kv_heads=2,  mlp=11008, layers=36, vocab=151936),
    "Qwen2.5-7B":  dict(embed=3584, head_dim=128, heads=28, kv_heads=4,  mlp=18944, layers=28, vocab=152064),
}


# --------------------------------------------------------------------------- #
# Compute model: prefill MAC counts                                            #
# --------------------------------------------------------------------------- #
def prefill_macs(model, n):
    """MACs for one prefill pass over `n` tokens (batch 1).

    Returns (linear, attention): `attention` is the KV-cache-touching matmuls
    (Q@K^T and S@V); `linear` is every other MAC (QKVO projections, gated MLP,
    lm_head). Element-wise ops (softmax, SiLU) are not MACs and not counted.
    """
    m = MODELS[model]
    E, L, M = m["embed"], m["layers"], m["mlp"]
    q_dim = m["heads"] * m["head_dim"]
    kv_dim = m["kv_heads"] * m["head_dim"]

    proj = E * q_dim + 2 * E * kv_dim + q_dim * E   # q, k, v, out projections
    gated_mlp = 3 * E * M                            # gate, up, down
    linear = L * n * (proj + gated_mlp) + n * E * m["vocab"]
    attention = L * n * n * (q_dim + kv_dim)
    return linear, attention


def weight_elems(model):
    """Weight elements loaded in one forward pass (projections + MLP + lm_head)."""
    m = MODELS[model]
    E, M = m["embed"], m["mlp"]
    q_dim = m["heads"] * m["head_dim"]
    kv_dim = m["kv_heads"] * m["head_dim"]
    per_layer = E * q_dim + 2 * E * kv_dim + q_dim * E + 3 * E * M
    return m["layers"] * per_layer + E * m["vocab"]


# --------------------------------------------------------------------------- #
# Energy model (Figure 16)                                                     #
# --------------------------------------------------------------------------- #
SEQ_LEN = 2048          # prefill length the energy numbers are reported at
T, To = 2048, 256       # tiling: sequence tile and output-channel tile
DRAM_PJ_PER_BYTE = 3.2e-11
SRAM_ACT_PJ = 2.428e-11
SRAM_PSUM_PJ = 2.92e-11
PE_LANES = 32 * 128     # 32 PEs x 128 MACs/cycle

# Per-architecture hardware parameters. pj_per_mac comes from PrimeTime power
# reports; bit widths are the effective (scale-inclusive) bits / 8 = bytes per
# element; Tk is the inner-tile depth; accum16 doubles psum SRAM traffic.
ARCHS = {
    #             pJ/MAC      act bytes  wgt bytes  kv bytes  Tk   accum16
    "Amove": dict(pj=5.04e-13,   act=4.25/8, wgt=4.25/8,  kv=5/8,     Tk=256, accum16=True),
    "MXFP":  dict(pj=1.50e-13,   act=8.25/8, wgt=4.25/8,  kv=None,    Tk=256, accum16=True),
    "NVFP4": dict(pj=1.68125e-13, act=4.5/8,  wgt=4.5/8,   kv=None,    Tk=256, accum16=True),
    "HBQ-A": dict(pj=1.336e-13,  act=5.19/8, wgt=4.25/8,  kv=None,    Tk=512, accum16=False),
    "HBQ-E": dict(pj=8.72e-14,   act=5.1/8,  wgt=4.125/8, kv=None,    Tk=512, accum16=False),
}

def dram_traffic(model, n=SEQ_LEN):
    """Element counts behind the DRAM energy terms (one prefill pass)."""
    m = MODELS[model]
    E, L, M = m["embed"], m["layers"], m["mlp"]
    kv_dim = m["kv_heads"] * m["head_dim"]
    return dict(
        wgt=weight_elems(model),
        qkvo=4 * L * n * E,               # Q/K/V/O activation tiles at width E
        attn_kv=4 * L * n * kv_dim,       # K,V written once + read once each
        mlp=L * n * (2 * E + 3 * M),      # gate/up inputs + gate/up/silu outputs
    )


def energy(arch, model):
    """Energy (J) per component for one prefill pass: dram / core / sram."""
    a = ARCHS[arch]
    m = MODELS[model]
    E, M = m["embed"], m["mlp"]
    heads, kv = m["heads"], m["kv_heads"]
    traffic = dram_traffic(model)

    linear, attention = prefill_macs(model, SEQ_LEN)
    total_macs = linear + attention
    core = a["pj"] * total_macs

    # Weights are re-streamed once per sequence tile of depth Tk.
    wgt = traffic["wgt"] * (T / a["Tk"]) * DRAM_PJ_PER_BYTE * a["wgt"]

    # Q/K/V/O tiles: each head group's share of the E-wide activation, per
    # output tile To; K/V are further scaled by the GQA ratio kv/heads.
    den = 2 * (heads + kv)
    q_size = o_size = (heads / den) * E / To
    k_size = v_size = (kv / den) * E * (kv / heads) / To
    qkvo = traffic["qkvo"] * (q_size + k_size + v_size + o_size) * DRAM_PJ_PER_BYTE * a["act"]

    # KV-cache / score tile traffic; Amove stores it at the KV-cache width.
    kv_bytes = a["kv"] if a["kv"] is not None else a["act"]
    attn_kv = traffic["attn_kv"] * (1024 / To) * DRAM_PJ_PER_BYTE * kv_bytes

    # MLP: the E-wide slice of the (2E + 3M) traffic, once per M/To tile.
    mlp = (E / (2 * E + M)) * (M / To) * traffic["mlp"] * DRAM_PJ_PER_BYTE * a["act"]

    dram = wgt + qkvo + attn_kv + mlp

    # SRAM: activation + partial-sum traffic per MAC lane; 16-bit accumulators
    # (Amove/MXFP/NVFP4) write psums at twice the width of HBQ's 8-bit ones.
    psum = SRAM_PSUM_PJ * (2 if a["accum16"] else 1)
    sram = total_macs / PE_LANES * (SRAM_ACT_PJ + psum)

    return dict(dram=dram, core=core, sram=sram, total=dram + core + sram)


def energy_table():
    """{(model, arch): energy dict} for every combination."""
    return {(mo, ar): energy(ar, mo) for mo in MODELS for ar in ARCHS}


# --------------------------------------------------------------------------- #
# Iso-area speedup model (Figure 18)                                           #
# --------------------------------------------------------------------------- #
# Area per MAC (um^2) from synthesis; throughput at iso-area scales inversely.
SCHEME_AREA = {
    "Amove": 217,
    "NVFP":  130.2,
    "MXFP":  101.3,
    "HBQ-A":  87,
    "HBQ-E":  72,
}
BASELINE_SCHEME = "Amove"

# The attention matmuls run at the activation width (> 4b), so a scheme's
# speedup is derated by 4/act_bits on the attention portion.
ATTENTION_DERATE = {
    "NVFP":  4 / 8.50,
    "MXFP":  4 / 8.25,
    "HBQ-A": 4 / 4.31,
    "HBQ-E": 4 / 4.13,
}

SPEEDUP_SEQ_LENS = {"1K": 1024, "4K": 4096, "16K": 16384}


def speedup_components(model, seq_key, scheme):
    """(linear, attention) portions of the stacked speedup bar."""
    linear_ops, attention_ops = prefill_macs(model, SPEEDUP_SEQ_LENS[seq_key])
    s_linear = SCHEME_AREA[BASELINE_SCHEME] / SCHEME_AREA[scheme]
    s_attention = s_linear * ATTENTION_DERATE.get(scheme, 1.0)

    time_linear = linear_ops / s_linear
    time_attention = attention_ops / s_attention
    time_total = time_linear + time_attention

    speedup = (linear_ops + attention_ops) / time_total
    return (speedup * time_linear / time_total,
            speedup * time_attention / time_total)


# --------------------------------------------------------------------------- #
# CLI: dump every computed number                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("== prefill MACs (linear / attention), batch 1 ==")
    for mo in MODELS:
        for key, n in [("1K", 1024), ("2K", 2048), ("4K", 4096), ("16K", 16384)]:
            lin, att = prefill_macs(mo, n)
            print(f"  {mo:<12} {key:>3}: linear={lin:>16}  attention={att:>16}")

    print("\n== DRAM traffic inputs (elements, prefill @2048) ==")
    for mo in MODELS:
        t = dram_traffic(mo)
        print(f"  {mo:<12} wgt={t['wgt']:>12}  qkvo={t['qkvo']:>12} "
              f"attn_kv={t['attn_kv']:>12}  mlp={t['mlp']:>12}")

    print("\n== energy (J), prefill @2048 ==")
    hdr = f"  {'model':<12} {'arch':<6} {'dram':>8} {'core':>8} {'sram':>8} {'total':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for (mo, ar), e in energy_table().items():
        print(f"  {mo:<12} {ar:<6} {e['dram']:>8.3f} {e['core']:>8.3f} "
              f"{e['sram']:>8.3f} {e['total']:>8.3f}")

    print("\n== iso-area speedup (linear + attention = bar height) ==")
    for seq in SPEEDUP_SEQ_LENS:
        print(f"  SeqLen={seq}")
        for mo in MODELS:
            parts = [f"{s}={sum(speedup_components(mo, seq, s)):.2f}" for s in SCHEME_AREA]
            print(f"    {mo:<12} " + "  ".join(parts))
