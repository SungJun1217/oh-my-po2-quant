# NPU / Integer-Inference Op Survey for Po2-Scale INT8 PTQ

Status: research note, 2026-09-26. Scope: how shipping NPUs and integer runtimes implement the "awkward" ops in INT8 inference (nonlinear activations, average pooling, residual add, concat, first and last layer, SE multiply, resize, max pool, channel shuffle). The goal is a set of **defensible defaults** for our target profile:

```text
INT8 W / INT8 A, symmetric, per-tensor, scale = 2^k
INT8 bias (configurable 8/16/32), INT32 accumulator
shift-only requantization
target NPU op semantics: UNKNOWN (to be confirmed with vendor)
```

Conventions used here:

- `k` is the Po2 exponent, so scale = 2^k. Vitis AI uses a **fix position** `p` with scale = 2^-p (so `p = -k`). Larger `p` means a finer scale.
- "Verified" means I read the source code or the official doc text myself (URLs given). **[UNVERIFIED]** means I could not read a primary source, or my reading is an interpretation.
- Code references point to the `master`/`main` branches as fetched on 2026-09-26.

Xilinx/AMD **Vitis AI DPU** is the closest analog to our target: power-of-two, symmetric, per-tensor quantization ("The quantization for DPU uses power-of-2 scales, symmetry, per-tensor quantizers and need some special processes to simulate DPU behaviors." — [vai_q_tensorflow2 README](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_tensorflow2.x/README.md)). Because it is the closest analog, this survey relies mainly on two DPU sources: its **quantizer** code, which records the compiler constraints, and its **CPU reference runner** (`vart/cpu-runner`), which is a bit-level golden model of the fixed-point ops.

---

## 0. Primary sources used

| Stack | Source actually read | Po2? |
|---|---|---|
| Vitis AI quantizer (PyTorch) | [`pytorch_nndct/nn/modules/*.py`](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/nn/modules) (avgpool, adaptive_avg_pool, mean, hardsigmoid, hardswish, sigmoid, leaky_relu, add, concat, multiply, interpolate) | yes |
| Vitis AI quantizer, compiler-constraint "refine" passes | [TF2 `vitis_pof2s_refine_transforms.py`](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/core/quantization/keras/vitis/quantize_strategy/pof2s/vitis_pof2s_refine_transforms.py), [ONNX `refine.py`](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_onnx/vai_q_onnx/refine.py) | yes |
| Vitis AI DPU golden model | [`vart/cpu-runner/src/op/`](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_runtime/vart/cpu-runner/src/op) (`eltwise_fix.cpp`, `pool_fix.cpp`, `concat_fix.cpp`, `conv2d_fix.cpp`, `hsigmoid_fix.cpp`, `resize_fix.cpp`, `reduce_mean_fix.cpp`), [`cpu_util.hpp` (DPURound)](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_runtime/vart/cpu-runner/include/cpu_util.hpp) | yes |
| DPUCZDX8G product guide (PG338) | [readkong mirror of PG338](https://www.readkong.com/page/dpuczdx8g-for-zynq-ultrascale-mpsocs-product-guide-6779141), [AMD PG338](https://docs.amd.com/r/en-US/pg338-dpu) | yes |
| TFLite / LiteRT | [int8 quantization spec](https://developers.google.com/edge/litert/models/quantization_spec), [`tfl_ops.td`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/ir/tfl_ops.td), [`kernels/add.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/add.cc), [`kernels/activations.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc), [`internal/common.h` (LUT)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.h), [`reference/integer_ops/pooling.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h), [`reference/hard_swish.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/hard_swish.h) | no (arbitrary scale); has a Po2 int16 add path |
| Arm Ethos-U / Vela | [`SUPPORTED_OPS.md`](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/SUPPORTED_OPS.md) (Vela 5.2.0), [`tflite_graph_optimiser.py`](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/ethosu/vela/tflite_graph_optimiser.py) | no |
| Arm CMSIS-NN | [`arm_avgpool_s8.c`](https://github.com/ARM-software/CMSIS-NN/blob/main/Source/PoolingFunctions/arm_avgpool_s8.c) | no |
| Apache TVM | [`relay/quantize/realize.cc`](https://github.com/apache/tvm/blob/v0.14.0/src/relay/quantize/realize.cc), [`relay/quantize/quantize.py`](https://github.com/apache/tvm/blob/v0.14.0/python/tvm/relay/quantize/quantize.py), [`relay/qnn/op/add.cc`](https://github.com/apache/tvm/blob/v0.14.0/src/relay/qnn/op/add.cc), [`concatenate.cc`](https://github.com/apache/tvm/blob/v0.14.0/src/relay/qnn/op/concatenate.cc) | `relay.quantize`: Po2 weights + shift realize |
| NVDLA | [Precision Preservation](https://nvdla.org/hw/v1/ias/precision.html), [Unit Description](https://nvdla.org/hw/v1/ias/unit_description.html), [LUT programming](https://nvdla.org/hw/v1/ias/lut-programming.html), [HW arch](https://nvdla.org/hw/v1/hwarch.html) | no (int multiplier + 5-bit shifter) |
| Google Edge TPU | [Coral: TensorFlow models on the Edge TPU](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/models-intro/) | no |
| Qualcomm QNN (via ONNX Runtime QNN quant config) | [`execution_providers/qnn/quant_config.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/execution_providers/qnn/quant_config.py) | no |
| Papers | Jacob et al. 2018, [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) (read via [ar5iv](https://ar5iv.labs.arxiv.org/html/1712.05877)); HAWQ-V3, [arXiv:2011.10680](https://arxiv.org/abs/2011.10680) (read via [ar5iv](https://ar5iv.labs.arxiv.org/html/2011.10680)) | HAWQ-V3: dyadic (b/2^c) |

Weakly verified or not covered: Qualcomm HTP op-def constraints (the [HTP Op Def Supplement](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/HtpOpDefSupplement.html) is rendered by JavaScript and I could not read it), Rockchip RKNN per-op semantics, NXP eIQ Neutron per-op constraints, Hailo defaults, and Intel Movidius (not researched). These are marked **[UNVERIFIED]** wherever they appear.

---

## 1. Background: how the Vitis AI DPU datapath works

Everything below uses these DPU semantics. Every value comes from `conv2d_fix.cpp`, `eltwise_fix.cpp` and the refine passes.

- **Conv/FC requant**: `shift_cut = p_in + p_w - p_out`, and the compiler requires `0 <= shift_cut <= 16`. The golden model computes `acc*2 + floor(2*bias*2^shift_bias)`, divides by `2^(shift_cut+1)`, and then applies `DPURound`.
- **Bias has its own fix position.** It is *not* forced to `S_x*S_w`. It is shifted into the accumulator with `shift_bias = p_in + p_w - p_bias`, and the constraint is `min(0, -(24-(8+shift_cut))) <= shift_bias <= 16` (refine `adjust_shift_bias`). This bears directly on our INT8-bias problem (AGENTS.md §15). The DPU lets the INT8 bias be *coarser* than the accumulator scale and left-shifts it, so a large bias does not force the W/A exponents.
- **Rounding**: `DPURound` saturates to the data type's min/max, rounds `x.5` toward +inf for negative values (`ceil`), and otherwise uses `std::round`. **The effective rule is round-half-up (toward +inf).** On the quantizer side, the vai_q_tensorflow2 defaults are `activation_round_mode=1 (HALF_UP)` and `weight_round_mode=bias_round_mode=0 (HALF_TO_EVEN)` ([README](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_tensorflow2.x/README.md)).
- **Calibration methods** available for `pof2s`: only `0 = Non_Overflow` (MinMax) and `1 = Min_MSE`. The defaults are `weight_method=1`, `activation_method=1` and `bias_method=0` (same README).

---

## 2. Nonlinear activations: SiLU, Hardswish, Sigmoid (plus ReLU6 and LeakyReLU)

### 2(a) How stacks handle them

| Stack | Sigmoid | Hardswish / Hardsigmoid | SiLU / Swish | LeakyReLU | ReLU6 |
|---|---|---|---|---|---|
| **Vitis AI DPU** | TF2 quantizer: `convert_sigmoid_to_hard_sigmoid=True` by default. Otherwise sigmoid "will be left unquantized and will be scheduled on CPU" ([README](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_tensorflow2.x/README.md)). Some newer targets have a `VitisSigmoid` with these compiler constraints: input pos in [0,15], output pos >= 7, `shift_sigmoid = 14 + p_in - p_out` in [0,31] (refine `adjust_vitis_sigmoid`). The PyTorch quantizer also ships a ~2048-entry sigmoid table for a 16-bit "FPGA/AIE" mode (`sigmoid.py`, `sigmoid_table.py`). **Which DPU variants run native sigmoid: [UNVERIFIED]** | **Native PWL, not a LUT.** `hsigmoid(x) = relu6(x+3) * 2731/16384`, where 2731/2^14 ≈ 1/6 (`hardsigmoid.py`). In hardswish, the inner hsigmoid is re-quantized to scale **2^-7** (`scale_inv=128`, range [-128,127]) and then multiplied by x (`hardswish.py`). The golden model computes `x*2731 + 3*2731*2^hsigmoid_in` then `>> shift_hsigmoid`, and hswish = `x*hsig >> shift_hswish` (`conv2d_fix.cpp`, `hsigmoid_fix.cpp`). Both can be fused into conv or eltwise as a "nonlinear" type | TF2 converts `swish` to hard-swish by default (README Table 4). There is also an `adjust_shift_swish` constraint for `mul(x, sigmoid(x))`: `shift_swish = p_in0 + p_in1 - p_out` in [0,15] (TF2 refine). A 2026 YOLO-on-DPU paper replaces SiLU with HardSwish because sigmoid is "unsupported or poorly quantized on the target DPU" ([arXiv:2607.13106](https://arxiv.org/pdf/2607.13106), read from a search snippet only; **[UNVERIFIED detail]**) | **Only alpha = 0.1**, which the quantizer converts to 26/256 = 0.1015625 (README; `leaky_relu.py`). The golden model does `tmp*26/256` before rounding. Any other alpha runs on CPU | Fused clamp. The golden model clamps at `6<<4` only if `p_out <= 4`; otherwise INT8 saturation already clips below 6 (`conv2d_fix.cpp`) |
| **TFLite int8** | 256-entry int8 to int8 LUT. The output scale is **fixed at 1/256 with zero_point -128** ([spec](https://developers.google.com/edge/litert/models/quantization_spec); enforced in `activations.cc`: `output->params.scale == 1./256`). The int16 path uses a 513-entry table with linear interpolation (upper 9 bits index, lower 7 bits interpolate; `common.h`) | Fixed-point int16 computation, not a LUT (`reference/hard_swish.h`). Input and output scales are free | Not a builtin op (becomes LOGISTIC + MUL) | Two multipliers (identity and alpha) **[not re-read]** | Folded into the requant clamp |
| **Ethos-U (Vela)** | Converted to a **256-entry LUT** for int8/uint8 (`convert_tanh_sigmoid_to_lut` → `convert_to_lut8`, `ix = range(-128,128)`) | Converted to a 256-entry LUT (`convert_hardswish_to_lut`) | Via LOGISTIC LUT + MUL | int8: **LUT**. int16 with equal scales: native. Otherwise MUL + MAX (`convert_lrelu`) | Native clamp |
| **NVDLA** | SDP LUT with **two tables: X/LE (65 entries, linear or exponential indexing) and Y/LO (257 entries, linear)**, with linear interpolation, and linear interpolation also used for out-of-range inputs ([LUT programming](https://nvdla.org/hw/v1/ias/lut-programming.html)). LUT RAM is "(65+257)*2 bytes" ([unit description](https://nvdla.org/hw/v1/ias/unit_description.html)) | Via LUT | Via LUT | PReLU is native in SDP X1/X2 | Native |
| **Edge TPU** | LOGISTIC supported, no limits listed ([Coral](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/models-intro/)) | HARD_SWISH support is **[UNVERIFIED]**: a search summary lists it, but the Coral table I fetched did not | n/a | [UNVERIFIED] | ReLU6 supported |
| **Qualcomm QNN** | ORT's QNN config overrides **16-bit** sigmoid output to scale 1/65536 (uint16, zp 0) or 1/32768 (int16, zp 0) "as per QNN requirements" (`quant_config.py`). The 8-bit rule is **[UNVERIFIED]** | [UNVERIFIED] | [UNVERIFIED] | [UNVERIFIED] | [UNVERIFIED] |
| **RKNN** | Changelog mentions fused Conv-Sigmoid / Conv-Swish / Conv-HardSwish ([rknn-toolkit2 CHANGELOG](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/CHANGELOG.md), via search summary). **Implementation (LUT or PWL) [UNVERIFIED]** | same | same | [UNVERIFIED] | [UNVERIFIED] |

**LUT indexing convention (TFLite, verified in `common.h`).** The int8 input code is cast to `uint8` and used directly as the index. The table is therefore ordered `[0..127, -128..-1]` and is filled offline:

`lut[q] = clamp(round(f(S_in*(q - zp_in)) / S_out) + zp_out)`

Every per-layer scale is baked into the table, so the LUT absorbs the requantization for free. Vela builds its tables the same way.

### 2(b) Implications under Po2-only scales

1. **A 256-entry LUT is scale-agnostic.** The table is regenerated per layer from `(k_in, k_out)`, so Po2 scales cost nothing extra and `k_out` can be chosen by the usual discrete search. The only hardware requirement is a programmable 256-byte table per layer (or per invocation). Keep the index range consistent with the INT8 range convention: if activations are [-127,127], entry `-128` is unused but must still be defined.
2. **Sigmoid output scale.** With symmetric signed INT8 and a Po2 scale, the natural choice is `k_out = -7`, which covers [0, 127/128]. Sigmoid(x) ≥ 0.996 then saturates at 0.992, and the sign bit is wasted, leaving about 7 effective bits. TFLite gets the full 8 bits through asymmetry (1/256, zp -128), which our profile forbids. If the hardware supports an **unsigned** output for the LUT, `k_out = -8` gains one bit. Vitis uses 2^-7 for its internal hsigmoid, and the `VitisSigmoid` output pos must be >= 7, which is consistent with this.
3. **Hard-sigmoid/hard-swish natively needs a 1/6 multiply.** 1/6 is not Po2. Vitis uses an integer multiply by 2731 followed by `>>14`. A strictly shift-only datapath would need a sum-of-shifts approximation such as `1/8 + 1/32 + 1/128 + ...` (**our construction, not seen in any stack**) or a small constant multiplier. A LUT avoids the question entirely.
4. **Replacing SiLU or sigmoid with hard variants changes the model.** This is the same situation as `/255 → /256`: it is a model transformation, and its FP32 loss must be measured and reported separately from PTQ loss (AGENTS.md §8).
5. **LeakyReLU:** alpha = 2^-n is exact with a shift. alpha = 0.1 needs either a LUT or an 8-bit multiply (26/256, as on the DPU).
6. **ReLU6 under Po2:** 6 is exactly representable when `6*2^-k_out <= 127`, i.e. `k_out >= -4`. For `k_out = -5` the output saturates at 3.97, which effectively turns ReLU6 into a clip at 3.97. This is a legitimate result of clipping-aware search, not a bug, and the DPU behaves the same way.

### 2(c) Recommendation

- **Default: `impl: lut`, 256 entries, INT8 in / INT8 out**, indexed by the raw INT8 input code (two's complement cast to uint8, TFLite-style). Tables are generated offline per layer from `(k_in, k_out)` using the **hardware rounding mode**. `k_out` is searched per layer, except sigmoid, where `k_out = -7` by default (unsigned output with `k_out = -8` if the hardware supports it). Apply this to SiLU, sigmoid, hardswish, hardsigmoid and LeakyReLU with alpha ≠ 2^-n. ReLU and ReLU6 remain native clamps fused into requant.
- **Alternative: `impl: pwl_native`** (DPU-style) for hardsigmoid/hardswish only: `relu6(x+3)*M >> s` with M/2^s ≈ 1/6 (Vitis: 2731/2^14), intermediate hsigmoid at 2^-7. SiLU and sigmoid must then be replaced by the hard variants, and that replacement is measured as a model-transformation loss.
  - Tradeoff: no LUT memory or programming cost, and more predictable. However, it needs a small constant multiplier (not pure shift), and SiLU-trained models (YOLOv5/v8, EfficientNet) lose FP accuracy unless fine-tuned.

### 2(d) Confirm with vendor

- Does a LUT exist? Size (256 entries? 512 with interpolation?), entry width (8 or 16 bit), per-layer reprogramming cost, index convention (signed cast or offset), and whether the LUT can be fused after conv requant.
- Is there any native PWL hsigmoid/hswish, and what are its constants (1/6 approximation, internal scale)?
- Can activation outputs be unsigned INT8?
- LeakyReLU: supported alpha values and how they are implemented.
- Exact rounding inside LUT generation or PWL arithmetic.

---

## 3. Average pooling / global average pooling with non-Po2 divisor (e.g. 49)

### 3(a) How stacks handle it

| Stack | Method | Details |
|---|---|---|
| **Vitis AI DPU** | **Sum, then an integer multiply by a hardware constant, then a shift.** In/out scales are forced equal | Fixed constants per kernel (`pool_fix.cpp`, `avgpool.py`, `dpu_utils.py`): 3x3 → **7/2^6**, 5x5 → **10/2^8**, 6x6 → **7/2^8**, 7x7 → **21/2^10**, 14x14 → **21/2^12**. Other sizes use `argmin_s |round(2^s/N)/2^s - 1/N|` over `s < ceil(log2(128N))`. The quantizer simulates this exact gain error when `nndct_avg_pool_approximate` is on. TF2 refine `align_pool` forces **input pos == output pos** for max and avg pool. `Mean` and `AdaptiveAvgPool` use the same factor search (`mean.py`, `reduce_mean_fix.cpp`). Vitis AI 3.5 can split large kernels into smaller ones ([release notes](https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/release_notes.html)). PG338 lists AvgPool kernel 1..256 as "always enabled" ([PG338 mirror](https://www.readkong.com/page/dpuczdx8g-for-zynq-ultrascale-mpsocs-product-guide-6779141)) |
| **TFLite reference / CMSIS-NN** | **Integer sum, then rounded integer division by count** | `acc = acc>0 ? (acc + n/2)/n : (acc - n/2)/n` (TFLite `pooling.h`; CMSIS-NN `arm_avgpool_s8.c`, identical). The spec and MLIR (`SameOperandsAndResultsScale`) require **input and output to share scale and zero point** |
| **Ethos-U (Vela)** | Hardware AvgPool with multiplier-based scaling. For stride > 3 it is rewritten as **Conv2D with all-ones weights and weight scale 1/(h*w)** (`convert_avg_pool_to_conv2d`). MEAN is rewritten as **DepthwiseConv2D, then MUL**, split into chunks when h*w > 4096 (`convert_mean_to_depthwise_conv`) | Constraints: VALID padding allows a kernel of up to 256x256, SAME padding up to 8x8 ([SUPPORTED_OPS](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/SUPPORTED_OPS.md)) |
| **NVDLA** | PDP multiplies by precomputed reciprocal scale factors: "pre_final_result × scale_factor_width × scale_factor_height" ([unit description](https://nvdla.org/hw/v1/ias/unit_description.html)) | Reciprocal precision is **[UNVERIFIED]** |
| **TVM `relay.quantize`** | Avg pool is computed in the **INT32 activation dtype** on the same dom-scale, with no division folded into the scale (`AvgPoolRealize`) | Requant happens later at the next `simulated_quantize` |
| **HAWQ-V3** | Pooling runs in **INT32** on the original feature map ("performing pooling on 4-bit can result in significant information loss") | ([ar5iv](https://ar5iv.labs.arxiv.org/html/2011.10680)) |

### 3(b) Implications under Po2-only scales

The divisor 1/N is a gain. A shift-only requant can realize only 2^-s, so a non-Po2 N always leaves a **residual gain** `g = 2^s/N`, unless the hardware has a small multiplier or the gain is moved into a place that can absorb it.

| N | DPU constant | gain error | best INT8 dw-weight `round(2^m/N) <= 127` | gain error | pure shift 2^-6 |
|---|---|---|---|---|---|
| 9 (3x3) | 7/64 | −1.56% | 114/2^10 | +0.20% | n/a |
| 25 (5x5) | 10/256 | −2.34% | 82/2^11 | +0.10% | n/a |
| 36 (6x6) | 7/256 | −1.56% | 114/2^12 | +0.20% | n/a |
| 49 (7x7) | 21/1024 | **+0.49%** | 84/2^12 (= 21/1024) | +0.49% | **−23.4%** |
| 196 (14x14) | 21/4096 | +0.49% | 84/2^14 | +0.49% | n/a |

(These are my computations from the constants above.)

- A gain error of ±0.5–2% is systematic across all channels. After BN folding the next layer is linear, so the error propagates as a scale error. The DPU accepts it and has the quantizer simulate it. Pure-shift division is unacceptable (−23% for 49) **unless the residual gain is compensated elsewhere**.
- **Headroom:** a sum of N INT8 values needs `8 + ceil(log2 N)` bits: N=49 → 14 bits; GAP over 20x20 (N=400) → 17 bits. This always fits in INT32. It does not fit in 16 bits for large GAP.
- **Fold trick (exact, graph-level):** avg = (sum >> s) * g with g = 2^s/N. If the pool output feeds a linear layer (the FC head, or the first FC/1x1 conv of an SE block), fold g into that layer's FP32 weights **before** weight quantization. g > 0 also passes through ReLU (positive homogeneity), but it does **not** pass through sigmoid, hswish, a residual add or a concat. This is a graph transformation, so an FP32 equivalence test is required (AGENTS.md §34). The downstream weight exponent must then be re-searched.
- **Depthwise-conv form:** an INT8 constant weight `q_w = round(2^m/N)` with Po2 weight scale 2^-m is literally an ordinary INT8 Po2 depthwise conv (bias 0) inside our existing datapath. Only the accumulator width and the requant shift matter. This is the same idea as Vela's ones-weight conv.

### 3(c) Recommendation

- **Default: `divisor: dwconv_int8_weight`.** Lower `AvgPool(N)` to a depthwise conv with constant weight `q_w = round(2^m/N)`, where m is the largest value with `q_w <= 127`, scale 2^-m, and no bias, followed by the standard shift requant to `k_out`. When N is Po2 (2x2, 4x4, 8x8) this reduces to an exact shift. The **simulator must model the actual `q_w/2^m` gain**, as Vitis does. Output exponent: search it. Do not blindly force `k_out = k_in` (TFLite, Vitis and Ethos-U force equality, but they have a multiplier).
- **Alternative: `divisor: shift_and_fold`.** Compute `sum >> s` (Po2 part), fold the residual gain `2^s/N` into the next linear layer's FP32 weights (exact in FP), and fall back to the default where the consumer is not linear.
  - Tradeoff: no gain error at all and no multiplier needed. However, it requires a graph rewrite plus an equivalence test, and it is limited to GAP → FC/Conv(+ReLU) patterns (these cover the classifier head and the SE squeeze).

### 3(d) Confirm with vendor

- Does the pooling engine have a multiplier? If so, are the constants fixed (DPU-style 7/64, 21/1024) or programmable, and at what precision?
- Can avg pool run as a depthwise conv with a large kernel (7x7, or HxW for GAP)? What are the maximum kernel size and accumulator width?
- Rounding of the final shift.
- Is in/out scale equality required?
- For GAP: the maximum reduction size before splitting.

---

## 4. Residual add with different input Po2 scales

### 4(a) How stacks handle it

| Stack | Alignment policy | Details |
|---|---|---|
| **Vitis AI DPU** | **Align to the coarsest input (min fix pos) with 2 guard bits, floor, then one write-shift with round-half-up** | `shift_read[i] = p_in[i] - min(p_in)`, `shift_write = min(p_in) - p_out` (`eltwise_fix.cpp`). Each input is computed as `floor(x*4 / 2^shift_read)`, which is **truncation with 2 fractional guard bits**. The sum is divided by `2^shift_write * 4`, then `DPURound` saturates. Constraints: `0 <= shift_read <= 15`, `-15 <= shift_write <= 15` (refine `adjust_shift_read/write`, for Add **and** Mul). PG338: eltwise sum supports **1–4 inputs**. Scales may differ, so there is **no forced equality** |
| **TFLite int8** | Rescale both inputs to `2*max(S1,S2)` after a **left shift of 20**, then an output multiplier | `left_shift = 20` (int8), `real_input_i_multiplier = S_i/(2*max S)`, `output_multiplier = 2*max S / (2^20 * S_out)` (`add.cc`). Input and output scales are all free. There is a separate **int16 Po2 path** (`pot_scale_int16`) with symmetric zp = 0, where one input may be right-shifted, "the graph quantization should ensure that the other input matches the output", and `input_shift <= 0` |
| **TVM QNN** | Requantize each input to the output scale, then an INT32 add | (`qnn/op/add.cc`, `RequantizeOrUpcast`) |
| **TVM `relay.quantize`** (Po2 flavor) | **Unify to the smaller (finer) scale**: `return s1 > s2 ? s2 : s1`. The coarser input is left-shifted in INT32, then the sum is right-shifted with a round bias (`round_for_shift=True`) | (`realize.cc: ChooseDomScale`, `MulAndDiv`) |
| **HAWQ-V3** | Each branch is scaled by its own dyadic number into INT32, the branches are added in **INT32**, and one requant follows | ([ar5iv](https://ar5iv.labs.arxiv.org/html/2011.10680)) |
| **Jacob 2018** | "one input needs to be rescaled onto the other's scale using a fixed-point multiplication … the result must be rescaled again to fit the output array's scale" | ([ar5iv](https://ar5iv.labs.arxiv.org/html/1712.05877)) |
| **NVDLA** | SDP eltwise SUM requires matching *scaling factors* on both input converters | ([precision](https://nvdla.org/hw/v1/ias/precision.html); summary-level reading, **[partially verified]**) |
| **Qualcomm HTP** | A fetch summary claimed equal-encoding requirements; the page is JS-rendered and I could not confirm it | **[UNVERIFIED]** |

### 4(b) Implications under Po2-only scales

- Alignment is a pure shift in both directions, so no multiplier is needed.
- **Aligning to the finer scale (left-shift the coarser input) is lossless**, provided the intermediate is wide enough. Two INT8 inputs with exponent gap `d` need `8 + d + 1` bits. INT16 covers a gap of up to about 7, INT32 any realistic gap.
- **Aligning to the coarser scale** is cheaper but drops `d` LSBs of the finer input. The DPU mitigates this with 2 guard bits, but `floor` introduces a negative bias of up to about 0.25 LSB (coarse) on each re-shifted input.
- The output exponent `k_out` is independent of both inputs and should be searched with clipping allowed. The sum range can be up to 2x the max input range, so blindly setting `k_out = max(k_in)` can saturate.
- Forcing `k1 = k2 = k_out` makes the add a trivial integer add plus saturation. However, it constrains two producers (and the skip branch's source) to one exponent. With Po2 granularity (2x steps) this is often harmless but occasionally costs one bit on one branch.

### 4(c) Recommendation

- **Default: `add.alignment: to_finer_exact`.** Left-shift the coarser input to `min(k1,k2)` in a wide (≥16-bit, preferably INT32) adder, add, then do **one** right-shift requant to a searched `k_out` using the hardware rounding mode and saturation. This is lossless before the final rounding and matches the TVM Po2 realize pass and the HAWQ-V3 INT32 add.
- **Alternative: `add.alignment: to_coarser_guard`** (DPU-style). Shift the finer input right to `max(k)` with `g` guard bits (DPU: g = 2, floor), then shift-write.
  - Tradeoff: narrower adder, but a small truncation bias. Needs `shift_read <= 15`, `|shift_write| <= 15` style limits.
- A second alternative is `equal_scales` (force producers to share an exponent). Use it only if the hardware cannot shift adder inputs.

### 4(d) Confirm with vendor

- Adder input width and whether each input can be shifted left, right or both, and by how much (DPU: 0..15 read, ±15 write).
- Rounding and floor behavior for the alignment shift and for the output shift.
- Saturation point (after the sum or after the output shift).
- Number of inputs supported (DPU: up to 4).
- Whether a fused ReLU or activation after the add is available.

---

## 5. Concat with different input scales

### 5(a) How stacks handle it

| Stack | Policy | Details |
|---|---|---|
| **Vitis AI DPU (quantizer)** | **Force all inputs and the output to the minimum fix pos (coarsest scale)**, rewriting producers' output positions | `align_concat`: `min_pos = min(p_out, p_in...)`, then every input's producer and the output are set to `min_pos` (TF2 refine and ONNX refine). The concat is then lossless data movement |
| **Vitis AI DPU (golden model)** | Hardware *can* shift per input: `shift_read[i] = p_out - p_in[i]`, `DPURound(x*2^shift)` (`concat_fix.cpp`) | The quantizer still aligns, so the per-input shift is normally 0 |
| **TFLite** | **Same scale and zero point** for all inputs and outputs (spec; MLIR `SameOperandsAndResultsScale` on `ConcatenationOp`) | Enforced by the converter's quantization pass |
| **Jacob 2018** | "all the input activations and the output activations in a Concatenation layer have the same quantization parameters … concatenations are thus lossless" | ([ar5iv](https://ar5iv.labs.arxiv.org/html/1712.05877)) |
| **TVM QNN** | Requantize each input whose params differ from the output, then concatenate (`concatenate.cc`). A FIXME in the code notes that requant-at-end would be better when all inputs match | n/a |
| **Edge TPU** | Supported, no fused activation. A constant input must be all zeros and there must be exactly 2 inputs ([Coral](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/models-intro/)) | Scale policy follows the TFLite model |
| **Ethos-U** | Only "32-bit feature-maps are not supported" is listed (SUPPORTED_OPS). Scale policy follows the TFLite model (equal) | n/a |

### 5(b) Implications under Po2-only scales

- The coarsest shared exponent introduces no new clipping (it covers the largest range) but costs resolution on small-range branches: `d` bits for an exponent gap `d`.
- A per-input requant by shift at concat time is cheap if the hardware supports it, but the rounding then happens twice (producer requant, then concat requant). The better approach is for **producers to requant directly into the shared exponent**. Conv producers already have a free shift, so constraining their output `k` costs nothing extra in hardware.
- Classic failure mode: **detection-head concat of box regression and class logits**. The two have very different ranges, and a shared exponent hurts one of them. Common deployment practice is to keep head branches as separate outputs. **[practice widely seen in vendor model zoos; no single citation verified]**

### 5(c) Recommendation

- **Default: `concat.policy: shared_scale_propagate`.** The shared exponent is chosen by searching over the union of the input distributions (clipping allowed, i.e. not simply `max(k_i)`), and every producer is forced to write at that exponent. This mirrors TFLite and Vitis `align_concat`, with the difference that Vitis always picks the coarsest exponent.
- **Alternative: `concat.policy: requant_inputs`.** Each producer keeps its own exponent, and the concat does a per-input shift to `k_out` (hardware permitting; DPU `concat_fix` has this).
  - Tradeoff: better per-branch fidelity for the producers, but an extra rounding step, and the hardware must be able to shift during concat or DMA.
- For detection heads: keep separate outputs instead of concatenating.

### 5(d) Confirm with vendor

- Is concat a zero-copy memory layout operation (which requires equal scales), or can the DMA/concat engine shift per input? If it can, what are the shift range and rounding?
- Maximum number of inputs and channel limits (DPU: output channels 1..256 × channel_parallel).

---

## 6. First layer and last layer precision

### 6(a) How stacks handle it

| Stack | First layer | Last layer |
|---|---|---|
| **Vitis AI DPU** | The input is **INT8 with a Po2 fix pos**. Preprocessing (mean/std) runs **on CPU in float**, then the result is multiplied by `2^fix_point` and cast. The VART sample does `(B - mean)*scale*fix_scale` with `fix_scale = exp2(fix_point)` ([`resnet50_pt.cpp`](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_runtime/resnet50_pt/resnet50_pt.cpp), [`common.h`](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_runtime/common/common.h)). The quantizer's `input_layers` argument can exclude early layers from quantization | INT8 by default. The `output_layers` argument can stop quantization before the last layer, which then runs on CPU. The output is dequantized with `2^-fix_point`. Softmax runs on CPU or a dedicated Softmax IP (README Table 4) |
| **TFLite** | Full int8, with optional int8/uint8 I/O | int8 logits (output dequantized) |
| **TVM `relay.quantize`** | Default `skip_conv_layers=[0]`: the **first conv is not quantized** (`quantize.py`) | Default `skip_dense_layer=True`: **dense layers are not quantized** |
| **HAWQ-V3** | "keeping the first and last layer in 8-bit", even in INT4 mixed-precision runs | same |
| **NVDLA** | A special **image-input convolution mode** for the first layer (3-channel image surfaces with channel extension) ([hwarch](https://nvdla.org/hw/v1/hwarch.html)) | n/a |
| **Hailo** | Per-layer `precision_mode` (e.g. `a16_w16`) is available, and users commonly apply 16-bit to output layers ([Hailo community](https://community.hailo.ai/t/how-to-apply-16-bits-quantization-to-all-the-convolution-layers-in-the-model/2837), [thread](https://community.hailo.ai/t/16-bit-quantization-on-final-layers/2292)). **Defaults [UNVERIFIED]** | same |
| **RKNN** | Toolkit1 offered `dynamic_fixed_point-8/16`, i.e. Po2 fixed point, plus hybrid (per-layer) quantization ([Firefly wiki](https://wiki.t-firefly.com/en/CORE-1126-JD4/rknn.html)). Per-layer defaults are **[UNVERIFIED]** | same |

### 6(b) Implications for our `/256` input and Po2 profile

- `x = p * 2^-8`, where p ∈ [0,255], **does not fit symmetric signed INT8** ([-127,127]). The options are:
  1. **Unsigned INT8 input at k = -8.** Exact, and zero padding stays correct. Requires a uint8 input path on the first conv.
  2. **`p >> 1` as signed INT8 at k = -7.** Symmetric and pads correctly, but drops one input LSB (7-bit image). Rounding must match the deployment path.
  3. **`p - 128` as signed INT8 with the offset folded into the first-layer bias** (`b' = b + 0.5*ΣW`). This is exact in the interior but **wrong at padded borders**: the padding value would have to be -128, not 0. It also inflates the INT8 bias range, which interacts with bias saturation (AGENTS.md §15).

  **This must be decided from hardware facts, not by the quantizer.**
- The first layer's weights see raw image statistics. Po2 per-tensor weight quantization of a 3-channel conv is usually fine, but it is the classic place where per-tensor scaling hurts (few channels, wide spread). If accuracy requires it, INT16 activations or weights for this layer are a mixed-precision *hardware* question.
- **Last layer:** logits and box regressions are sensitive to requantization. The cheapest improvement is to **output the INT32 accumulator (or an INT16 requant) instead of INT8**, and leave final dequantization to the host.

### 6(c) Recommendation

- **Default:** first and last layers are **INT8 W / INT8 A**, same as the rest (Vitis, TFLite, HAWQ-V3 norm). The first-layer input is **UINT8 at k = -8 (exact `/256`)** if the hardware allows; otherwise the fallback is `p>>1` at k = -7 (option 2), treated as part of the preprocessing contract and reflected in Baseline B. The last layer writes INT8 by default, with `last_layer.output: int32_acc` enabled if the hardware can write wide outputs.
- **Alternative:** INT16 activations on the first input and/or the last output (Hailo-style per-layer 16-bit).
  - Tradeoff: better fidelity where PTQ loss concentrates, but it requires 16-bit datapath or DMA support and breaks uniform INT8.

### 6(d) Confirm with vendor

- Is an unsigned 8-bit input supported for the first layer? What is the padding value semantics?
- Is there an image-input mode (NVDLA-like) or a hardware preprocessing block (mean subtraction, Po2 scale)?
- Can outputs be INT16 or INT32 (raw accumulator)?
- Is per-layer mixed precision supported at all?

---

## 7. Other ops (brief)

### 7.1 SE-block multiply (sigmoid gate × feature)

- **Vitis DPU:** eltwise MUL has `shift_write = Σp_in - p_out`. The golden model computes `Π floor(x*4/2^shift_read)` then `/4^n` (`eltwise_fix.cpp`), and the refine passes constrain the MUL shift ranges. The gate is typically hard-sigmoid (sigmoid is converted by default). PG338: eltwise multiply takes exactly 2 inputs.
- **TFLite:** MUL takes arbitrary scales with one output multiplier (spec). The logistic gate output is at 1/256.
- **Po2:** the product exponent is `k_x + k_g`, and requant is a right shift by `k_out - (k_x + k_g)`, so no multiplier is needed. With gate ∈ [0,1] at `k_g = -7`, `k_out = k_x` is a natural starting point, because the output range is ≤ the input range. Search `k_out` anyway (finer is often better, since gated values shrink).
- **Recommend:** default gate at signed INT8 `k = -7` (unsigned `k = -8` if available), an INT16 or INT32 product, and a shift requant to a searched `k_out`. Alternative: fold the gate into a per-channel scale. That is not allowed here, because it implies per-channel dynamic scaling. **Confirm** that MUL exists and what its product width and shift range are.

### 7.2 Upsample (nearest / bilinear)

- **Nearest:** pure copy. TFLite requires the same scale (`SameOperandsAndResultsScale`). The Vitis golden `upsample-fix` allows a shift `p_out - p_in` with DPURound (`resize_fix.cpp`), and Vitis quantizes the interpolate output separately (`interpolate.py`). Ethos-U supports factors 2x/4x/8x (SUPPORTED_OPS).
- **Bilinear:** TFLite requires the same scale. Ethos-U requires 2x with half-pixel centers, or 2x/4x/8x otherwise, and lowers it to depthwise convs (`convert_resizebilinear_to_depthwise_convolutions`). The Edge TPU "might not be mapped … to avoid loss in precision".
- **Po2:** a 2x half-pixel bilinear has weights 0.25/0.75, which are exact with 2 fractional bits. Arbitrary factors are not Po2-friendly.
- **Recommend:** nearest with `k_out = k_in` as the default. Bilinear only for 2x/4x/8x Po2 factors, computed with ≥2 guard bits and one final rounding. **Confirm** the supported modes and factors, and the rounding.

### 7.3 Max pool

- Order-preserving, so it is exact when `k_out = k_in`. Required by TFLite (spec), Vitis `align_pool` (forces input pos == output pos), and Ethos-U MAXIMUM ("Both Input quantization parameters must match OFM").
- **Recommend:** force `k_out = k_in`. Propagate the shared exponent across ReLU/max-pool chains.

### 7.4 Channel shuffle / reshape / transpose / pixel shuffle

- Pure data movement. TFLite requires the same scale for TRANSPOSE, RESHAPE and similar ops (spec). Vitis has `pixel_shuffle`/`reorg` ops with `fix` variants (cpu-runner op list).
- **Recommend:** treat as scale-transparent (`k_out = k_in`), with no requant. **Confirm** that the hardware executes them without an implied requant (some compilers lower shuffle to a 1x1 conv, which would add weight-quant error). **[UNVERIFIED for any specific NPU]**

---

## 8. Cross-cutting findings to carry into the hardware profile

1. **Rounding differs by stack.** DPU: round-half-up (toward +inf), with floors inside eltwise alignment. TFLite: the reference uses round-half-away for division and gemmlowp rounding-doubling multiplies. TVM Po2 realize: add `2^(s-1)` then arithmetic right shift. Do not reuse any of these without vendor confirmation (AGENTS.md §23).
2. **Shift ranges are bounded in real hardware.** DPU: `shift_cut ∈ [0,16]`, `shift_bias ∈ [min(0, shift_cut - 16), 16]`, `shift_read ∈ [0,15]`, `shift_write ∈ [-15,15]`. The exponent search must respect such bounds, which means the search space is constrained, not free.
3. **INT8 bias on a Po2 NPU (DPU):** the bias has its own exponent and is left-shifted into the accumulator. This is a concrete precedent for relaxing `S_b = S_x*S_w` when bias is INT8. It is worth a separate study against AGENTS.md §13–16, where the target hardware would need to support a `bias_shift`.
4. **Every multiplier-free design still hides small integer multipliers** somewhere: DPU avg pool (7, 10, 21), hsigmoid (2731), LeakyReLU (26). A true shift-only NPU must replace these with a LUT, a fold, or a depthwise-conv weight. That is why the defaults below prefer LUT and dwconv lowering.

---

## 9. Proposed `operators:` section for the hardware profile

Values marked `# UNVERIFIED-HW` are our defaults pending vendor confirmation. Field names are chosen to map one-to-one onto simulator switches.

```yaml
operators:
  # ---- global semantics that op policies reference ----
  rounding_mode: round_half_up          # UNVERIFIED-HW (DPU=half-up; alternatives: half_even, half_away, floor)
  saturation: saturate                  # UNVERIFIED-HW
  shift_limits:                         # UNVERIFIED-HW; DPU values shown as reference
    requant_right_shift: [0, 16]        # DPU shift_cut
    bias_left_shift: [0, 16]            # DPU shift_bias upper bound (bias exponent may differ from kx+kw)
    eltwise_input_shift: [0, 15]        # DPU shift_read
    eltwise_output_shift: [-15, 15]     # DPU shift_write

  activation:
    relu:        {impl: native_clamp}
    relu6:       {impl: native_clamp, allow_clip_below_6: true}   # k_out search may clip < 6
    leaky_relu:
      impl: lut                          # alt: native_mul (alpha quantized to n/256, DPU: 26/256)
      alpha_po2_shortcut: true           # alpha == 2^-n -> native shift, no LUT
    sigmoid:
      impl: lut                          # alt: replace_with_hard_sigmoid (model-transform loss, report separately)
      output_exponent: -7                # symmetric int8 -> [0, 127/128]; -8 if unsigned output supported
    hardsigmoid:
      impl: lut                          # alt: pwl_native (relu6(x+3)*M>>s, M/2^s≈1/6; DPU 2731/2^14)
      pwl_inv6_multiplier: null          # set e.g. {m: 2731, shift: 14} only if HW has it
    hardswish:
      impl: lut                          # alt: pwl_native with internal hsigmoid exponent -7 (DPU)
      pwl_internal_exponent: -7
    silu:
      impl: lut                          # alt: replace_with_hardswish (model-transform loss)
    lut:
      entries: 256                       # UNVERIFIED-HW
      index: int8_code_as_uint8          # TFLite/Vela convention: lut[uint8(q)]
      in_bits: 8
      out_bits: 8
      out_signed: true                   # UNVERIFIED-HW; unsigned doubles sigmoid resolution
      generated_offline_per_layer: true  # table absorbs (k_in -> k_out); k_out searched
      interpolation: none                # alt: 16-bit index w/ linear interp (TFLite int16: 513 entries; NVDLA 65+257)

  avgpool:
    divisor_handling: dwconv_int8_weight # q_w = round(2^m/N) <= 127, weight exp -m; exact for Po2 N
    alternative: shift_and_fold          # sum>>s, fold 2^s/N into next linear layer (FP32 equivalence test required)
    simulate_gain_error: true            # model q_w/2^m (e.g. 49 -> 84/4096, +0.49%) exactly
    output_exponent: search              # not forced equal to input (we have no multiplier to keep it exact)
    accumulator_bits: 32
    max_kernel: null                     # UNVERIFIED-HW
  global_avgpool:
    lower_to: avgpool                    # same policy; split if reduction > HW limit
    max_reduction: null                  # UNVERIFIED-HW

  add:
    alignment: to_finer_exact            # left-shift coarser input to min(k); alt: to_coarser_guard, equal_scales
    guard_bits: 2                        # used only by to_coarser_guard (DPU uses 2, floor)
    intermediate_bits: 32                # UNVERIFIED-HW (>= 8 + max_exp_gap + 1)
    output_exponent: search              # clip-aware; not forced to max(k_in)
    max_inputs: 2                        # DPU supports up to 4
  mul:                                   # SE gate x feature, and x*sigmoid(x)
    product_bits: 16                     # UNVERIFIED-HW
    requant: shift                       # shift = k_out - (k_a + k_b)
    gate_exponent: -7
    output_exponent: search

  concat:
    policy: shared_scale_propagate       # alt: requant_inputs (per-input shift at concat)
    shared_exponent_selection: search_union   # Vitis uses min fix pos (coarsest); we search
    split_detection_heads: true          # keep box/cls outputs separate instead of concatenating

  maxpool:             {scale_policy: equal_in_out}
  upsample_nearest:    {scale_policy: equal_in_out, factors: [2, 4, 8]}
  upsample_bilinear:   {supported: po2_factors_only, guard_bits: 2, scale_policy: equal_in_out}   # UNVERIFIED-HW
  data_movement:       # reshape, transpose, channel_shuffle, pixel_shuffle, slice, pad
    scale_policy: equal_in_out
    pad_value_semantics: zero_code       # must be revisited if input uses an offset (p-128) encoding

  first_layer:
    input_encoding: uint8_k_minus_8      # exact image/256; alt: int8_shr1_k_minus_7 (p>>1); avoid p-128 + bias fold (padding mismatch)
    weight_bits: 8
    activation_bits: 8
    allow_int16_activation: false        # UNVERIFIED-HW mixed precision
  last_layer:
    weight_bits: 8
    activation_bits: 8
    output: int8                         # alt: int32_acc (raw accumulator) or int16 if HW can write wide outputs
    allow_int16_activation: false        # UNVERIFIED-HW
```

---

## 10. Open questions for the vendor (consolidated)

1. Rounding mode(s) for requant, eltwise alignment and LUT generation. Saturation point(s).
2. Shift ranges: requant, bias left-shift, eltwise read/write. Does bias use `S_x*S_w` exactly, or a separate exponent plus shift (DPU-style)?
3. LUT: exists? Entries, width, index convention, reprogramming cost, fusion with conv.
4. Any native PWL hsigmoid/hswish? Any small constant multipliers anywhere (pool, activation)?
5. Avg pool: multiplier constants (fixed or programmable), max kernel, depthwise-conv lowering, GAP reduction limit.
6. Add: input shift directions and range, adder width, number of inputs, fused activation.
7. Concat: zero-copy (equal scales required) or per-input shift?
8. First layer: unsigned input support, padding semantics, preprocessing block.
9. Last layer: INT16/INT32 output support. Per-layer mixed precision.
10. MUL (SE), resize modes and factors, channel shuffle lowering.
