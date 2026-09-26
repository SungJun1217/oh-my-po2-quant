# AGENTS.md

# all-you-need-is-po2-quant

## 1. Project Goal

This repository researches and implements **hardware-friendly Post-Training Quantization (PTQ)** under strict **Power-of-2 scale constraints**.

The main target is:

```text
INT8 Weight
INT8 Activation
Power-of-2 Scale
Symmetric Quantization
Primarily Per-Tensor Quantization
Configurable Bias Precision
```

The core research question is:

> Under strict hardware constraints, how can we select Power-of-2 quantization scales that minimize real model accuracy degradation?

The project prioritizes:

```text
1. Hardware compatibility
2. End-to-end task accuracy
3. Layer-output fidelity
4. Tensor reconstruction error
```

Do not optimize proxy metrics at the expense of actual model accuracy.

---

# 2. Definition of Power-of-2 Quantization

In this repository, **Power-of-2 quantization means the quantization scale is constrained to powers of two**.

For tensor `x`:

```text
scale = 2^k
```

Quantization:

```text
q = clip(round(x / scale), qmin, qmax)
```

Dequantization:

```text
x_hat = q * scale
```

For symmetric INT8, normally assume:

```text
q ∈ [-127, 127]
```

unless the target implementation explicitly uses:

```text
[-128, 127]
```

Always verify the real hardware/compiler behavior.

This project is **NOT** about logarithmic quantization where individual weight values themselves are restricted to powers of two.

Do not confuse the two.

---

# 3. Why Po2 Scaling Exists

The primary motivation is hardware efficiency.

An arbitrary scale such as:

```text
0.03721
```

requires multiplication or fixed-point rescaling.

A Power-of-2 scale such as:

```text
2^-5 = 0.03125
```

can conceptually be implemented using shift-friendly arithmetic.

Example:

```text
x * 2^-5
```

can map to:

```text
x >> 5
```

or equivalent hardware fixed-point logic.

Therefore arbitrary floating-point scales must never be introduced unless explicitly supported by the target hardware.

---

# 4. Po2 Scale Selection Is a Discrete Optimization Problem

Do not treat Po2 scale selection as:

```text
1. Find arbitrary floating-point scale
2. Round it once to nearest Power-of-2
```

That is only a baseline.

Example naïve method:

```python
scale_fp = absmax / 127
k = round(log2(scale_fp))
scale = 2**k
```

The better formulation is:

```text
k* = argmin_k Error(Q(x, 2^k))
```

where `k` is searched over discrete candidates.

A typical search may use:

```text
base_k - 4
...
base_k
...
base_k + 4
```

or a wider configurable range.

Always evaluate actual candidate scales.

---

# 5. Clipping vs Resolution

Selecting a Po2 exponent creates a tradeoff between:

```text
clipping error
vs
rounding / quantization error
```

Example tensor:

```text
range ≈ [-3.4, 3.4]
```

Candidate:

```text
scale = 2^-5 = 0.03125
```

Representable maximum:

```text
127 * 0.03125 = 3.96875
```

Little clipping, but lower resolution.

Candidate:

```text
scale = 2^-6 = 0.015625
```

Representable maximum:

```text
127 * 0.015625 = 1.984375
```

Better resolution, but strong clipping.

Therefore:

> Zero clipping is not necessarily optimal.

Do not automatically choose a no-clipping scale.

---

# 6. Calibration Must Match Deployment

This is one of the most important rules in the repository.

The calibration pipeline must reproduce the input distribution that the real hardware will see.

If deployment uses:

```text
image / 256
```

then calibration must also use:

```text
image / 256
```

Do not calibrate using `/255` and deploy using `/256`.

---

# 7. `/255` vs `/256`

Many models are originally trained using:

```text
image / 255
```

However:

```text
1 / 255
```

is not a Power-of-2 value.

For Po2 hardware, deployment may instead use:

```text
image / 256
```

because:

```text
1 / 256 = 2^-8
```

Example:

```text
255 / 255 = 1.0
255 / 256 = 0.99609375
```

This difference is small, but it changes the network input distribution.

Therefore `/255 -> /256` must be treated as a real model/input transformation, not merely a quantization implementation detail.

---

# 8. Separate Preprocessing Loss From PTQ Loss

Always distinguish between:

```text
preprocessing approximation loss
```

and:

```text
quantization loss
```

Required baselines:

## Baseline A

```text
FP32
input = image / 255
```

Original model accuracy.

## Baseline B

```text
FP32
input = image / 256
```

Measures preprocessing approximation.

## Baseline C

```text
Po2 INT8
input = image / 256
```

Measures quantized deployment behavior.

Compute:

```text
Preprocessing Loss
=
Accuracy(FP32 /255)
-
Accuracy(FP32 /256)
```

Actual PTQ loss:

```text
PTQ Loss
=
Accuracy(FP32 /256)
-
Accuracy(INT8 Po2 /256)
```

Total deployment loss:

```text
Total Loss
=
Accuracy(FP32 /255)
-
Accuracy(INT8 Po2 /256)
```

Never report total degradation as pure PTQ degradation.

---

# 9. Input Preprocessing Is Part of the Quantization Contract

Treat preprocessing as part of the deployed quantized model.

This includes:

```text
resize
crop
letterbox
color conversion
channel order
normalization
mean subtraction
std normalization
input scaling
```

Example deployment path:

```text
uint8 image
    ↓
resize
    ↓
x * 2^-8
    ↓
quantized network
```

Calibration must reproduce the same path.

---

# 10. Mean/Std Normalization

A model may expect:

```text
(x / 255 - mean) / std
```

This cannot automatically be represented exactly using Po2 scaling.

Possible solutions include:

```text
1. Fold preprocessing into the first layer
2. Fold input scale into first-layer weights
3. Fold offsets into bias
4. Approximate scales using Po2 values
5. Use dedicated hardware preprocessing
```

Any transformation must first be verified in floating point.

Always compare:

```text
original FP32 output
vs
transformed FP32 output
```

before quantizing.

---

# 11. Folding Must Be Mathematically Correct

Given:

```text
x' = a*x + b
y  = W*x' + bias
```

then:

```text
y = W*(a*x + b) + bias
```

therefore:

```text
W'    = W*a
bias' = W*b + bias
```

However, real convolution requires correct handling of:

```text
tensor layout
channels
broadcasting
padding
group convolution
depthwise convolution
bias layout
```

Never perform folding based only on intuition.

Add numerical equivalence tests.

---

# 12. Weight and Activation Quantization

Weights and activations have different characteristics.

Weights:

```text
static
fully available offline
easy to search exhaustively
```

Activations:

```text
data-dependent
require representative calibration samples
distribution varies by layer
```

Always analyze them separately before performing joint optimization.

For every tensor, record where practical:

```text
tensor name
tensor type
min
max
absmax
mean
std
selected exponent
selected scale
clipping ratio
MSE
normalized MSE
cosine similarity
SQNR
```

---

# 13. Bias Is a First-Class Quantization Constraint

Bias must not be treated as an implementation afterthought.

For:

```text
x ≈ qx * Sx
w ≈ qw * Sw
```

the multiplication accumulator scale is:

```text
Sacc = Sx * Sw
```

Bias should therefore normally use:

```text
Sb = Sx * Sw
```

and:

```text
qb = round(b / (Sx * Sw))
```

For Po2:

```text
Sx = 2^kx
Sw = 2^kw
```

therefore:

```text
Sb = 2^(kx + kw)
```

This is naturally Power-of-2 compatible.

---

# 14. Bias Precision Must Be Configurable

Do not hard-code bias to INT32.

Support the concept of:

```text
bias_bits = 8
bias_bits = 16
bias_bits = 32
```

because different hardware implementations may have different restrictions.

Typical configuration:

```yaml
activation:
  bits: 8
  symmetric: true
  granularity: per_tensor
  scale: power_of_two

weight:
  bits: 8
  symmetric: true
  granularity: per_tensor
  scale: power_of_two

bias:
  bits: 8   # or 16 / 32
  scale: input_scale * weight_scale
```

Treat bias bit-width as a hardware constraint.

---

# 15. INT32 Bias vs INT8 Bias

INT32 bias is substantially less restrictive than INT8 bias.

Given:

```text
Sx * Sw = 2^-12
bias = 0.3
```

then:

```text
qb ≈ 0.3 * 4096
   ≈ 1229
```

INT32 can represent this directly.

INT8 cannot.

INT8 bias only allows approximately:

```text
[-127, 127]
```

Therefore INT8 bias may force different Weight/Activation exponents.

This means:

```text
optimal(INT8 bias)
!=
optimal(INT32 bias)
```

An INT8-bias optimized model may work well with INT32 bias, but the INT32 optimum should still be independently searched.

Do not assume that one optimum automatically transfers to another bias precision.

---

# 16. Bias Must Participate in Scale Optimization

When bias is restricted, Po2 scale optimization becomes a joint problem.

Do not optimize:

```text
Weight scale
Activation scale
```

independently while ignoring bias.

Instead model the actual integer datapath.

Conceptually:

```text
Xq = Q(X, 2^kx)
Wq = Q(W, 2^kw)

Bq = Qbias(B, 2^(kx + kw))

Yq = ConvInteger(Xq, Wq) + Bq
```

Then optimize:

```text
(kx*, kw*)
=
argmin_(kx, kw)
Error(Yfp, Yq)
```

subject to the target bias range.

For INT8 bias:

```text
Bq ∈ [-127, 127]
```

or the exact target hardware range.

---

# 17. Actual Integer Datapath Matters

The simulator should reproduce the real hardware path as closely as possible.

For Conv/Linear, reason about:

```text
INT8 input
×
INT8 weight
↓
integer multiply
↓
wide accumulator
↓
quantized bias add
↓
requantization
↓
INT8 output
```

Do not simulate only fake-quantized floating-point tensors if the resulting behavior differs from the hardware integer datapath.

Important hardware properties include:

```text
input precision
weight precision
bias precision
accumulator width
accumulator saturation
requantization method
shift behavior
rounding
output saturation
```

---

# 18. Accumulator Precision Is Separate From Bias Precision

Do not confuse:

```text
bias_bits
```

with:

```text
accumulator_bits
```

A hardware design may use:

```text
INT8 activation
INT8 weight
INT8 bias storage
INT32 accumulator
```

These are separate constraints.

Represent them independently.

Example:

```yaml
accumulator:
  bits: 32

bias:
  bits: 8
```

---

# 19. Bias Experiments

At minimum, support comparisons such as:

| Weight / Activation | Bias | Purpose |
|---|---:|---|
| INT8 Po2 | FP32 | Bias quantization reference |
| INT8 Po2 | INT32 | TFLite-style loose bias constraint |
| INT8 Po2 | INT16 | Intermediate constraint |
| INT8 Po2 | INT8 | Strict hardware constraint |

Use identical:

```text
model
calibration set
preprocessing
evaluation set
weight/activation constraints
```

when comparing bias precision.

---

# 20. Calibration Methods

The project should support progressively stronger methods.

## Level 0 — MinMax Po2

Example:

```text
scale_fp = absmax / 127

k = ceil(log2(scale_fp))

scale = 2^k
```

This is a baseline.

---

## Level 1 — MSE / OMSE Po2

Search discrete exponents:

```python
for k in candidates:
    scale = 2**k

    q = quantize(x, scale)
    x_hat = dequantize(q, scale)

    loss = mse(x, x_hat)
```

Choose the best candidate.

This should normally outperform naïve MinMax.

---

## Level 2 — Clipping-Aware Po2

Allow intentional clipping.

Possible metrics:

```text
MSE
NMSE
MAE
cosine similarity
SQNR
percentile objective
KL divergence
weighted MSE
```

Never assume MinMax is optimal.

---

## Level 3 — Layer Reconstruction

Prefer optimizing actual layer output when possible.

Example:

```text
Yfp = Conv(Xfp, Wfp, Bfp)

Yq = QuantizedConv(Xq, Wq, Bq)
```

Optimize:

```text
MSE(Yfp, Yq)
```

or:

```text
1 - cosine(Yfp, Yq)
```

Layer-output optimization may produce different scale choices than tensor MSE.

---

## Level 4 — Joint W/A/B Optimization

Search:

```text
activation exponent
weight exponent
bias representation
```

together.

Objective:

```text
argmin_(kx, kw) Error(
    Yfp,
    QuantizedLayer(X, W, B, kx, kw)
)
```

while respecting:

```text
activation range
weight range
bias range
accumulator behavior
output range
```

This is one of the main research directions of the project.

---

## Level 5 — Sensitivity / Second-Order Methods

Advanced techniques may include:

```text
Hessian-aware weighting
Fisher approximations
output sensitivity
cross-layer sensitivity
loss-aware quantization
```

Conceptually:

```text
ΔL ≈ 1/2 * Δw^T H Δw
```

However, mathematical sophistication alone is not sufficient.

Methods must work under:

```text
per-tensor
symmetric
INT8
Po2 scale
real bias constraint
real hardware datapath
```

---

# 21. Candidate Exponent Search

Do not restrict candidates unnecessarily.

A reasonable default may be:

```python
base_k = round(log2(reference_scale))

candidate_k = range(base_k - 4, base_k + 5)
```

but the search range must be configurable.

For Weight/Activation joint search:

```text
for kx in activation_candidates:
    for kw in weight_candidates:
        evaluate(kx, kw)
```

Since the search space is discrete and usually small, direct search is often practical.

Use pruning only when needed.

---

# 22. Record All Candidate Results

Do not store only the winning scale.

For each candidate, record where practical:

```text
exponent
scale
clipping ratio
MSE
NMSE
MAE
cosine similarity
SQNR
bias saturation ratio
layer output error
task accuracy delta
```

This is critical for debugging and research.

---

# 23. Rounding Must Match Hardware

Possible rounding behavior includes:

```text
round-half-to-even
round-half-away-from-zero
truncate
floor
hardware-specific fixed-point rounding
```

Do not assume:

```text
Python round
NumPy round
PyTorch round
```

matches hardware.

Implement target-specific rounding when necessary.

Add explicit boundary tests around:

```text
n + 0.5
```

---

# 24. Saturation Must Match Hardware

Check exact integer ranges.

Possible INT8 ranges:

```text
[-127, 127]
```

or:

```text
[-128, 127]
```

Do not assume.

Also verify saturation behavior for:

```text
weights
activations
bias
accumulator
requantization output
```

---

# 25. Requantization Must Be Modeled Correctly

Quantized layers usually require converting accumulator scale to output scale.

If:

```text
Sacc = Sx * Sw
```

and output scale is:

```text
Sy
```

then conceptually:

```text
qy ≈ qacc * Sacc / Sy
```

Under Po2:

```text
Sacc / Sy
=
2^(kx + kw - ky)
```

which can be implemented by shift.

The simulator must reproduce:

```text
shift direction
shift amount
rounding
saturation
```

used by hardware.

Do not ignore requantization error.

---

# 26. Quantization Optimization Should Follow the Real Layer

For a quantized layer, think in terms of:

```text
Input Quantization
      ↓
Integer MAC
      ↓
Bias Add
      ↓
Requantization
      ↓
Output Saturation
```

Do not independently optimize components if the combined behavior is what determines output quality.

The long-term optimization target is the complete layer datapath.

---

# 27. Calibration Dataset

Calibration data should approximate production data.

Avoid:

```text
random noise
synthetic unrelated images
very small unrepresentative sets
```

Record:

```text
dataset identity
number of samples
sample selection method
resize
crop
letterbox
normalization
random seed
```

Calibration must be reproducible.

---

# 28. Evaluation Must Be Task-Level

Tensor metrics are debugging tools.

Final evaluation must use the real task metric.

Examples:

Classification:

```text
Top-1
Top-5
```

Detection:

```text
mAP
AP50
AP75
```

Segmentation:

```text
mIoU
```

Other tasks:

Use their native production metric.

Never claim one quantization method is better solely because it has lower tensor MSE.

---

# 29. Required Experimental Baselines

Whenever possible compare:

```text
FP32 /255
FP32 /256
MinMax Po2
MSE Po2
Clipping-aware Po2
Layer reconstruction Po2
Joint W/A/B Po2
New method
```

If bias precision is relevant, run separately for:

```text
INT8 bias
INT16 bias
INT32 bias
```

Do not compare methods using different calibration data or preprocessing.

---

# 30. Preferred Result Table

Example:

```text
Method          Input   W/A         Bias   Metric   Delta
----------------------------------------------------------
FP32 Original   /255    FP32        FP32   xx.xx    -
FP32 Deploy     /256    FP32        FP32   xx.xx    -a
MinMax Po2      /256    INT8 Po2    INT32  xx.xx    -b
MSE Po2         /256    INT8 Po2    INT32  xx.xx    -c
Joint W/A/B     /256    INT8 Po2    INT8   xx.xx    -d
New Method      /256    INT8 Po2    INT8   xx.xx    -e
```

Also report separately:

```text
Preprocessing Loss
PTQ Loss
Bias Quantization Loss
Total Deployment Loss
```

where possible.

---

# 31. Hardware Configuration Must Be Explicit

Do not scatter hardware assumptions throughout code.

Use an explicit configuration.

Example:

```yaml
quantization:
  activation:
    bits: 8
    signed: true
    symmetric: true
    granularity: per_tensor
    scale_constraint: power_of_two

  weight:
    bits: 8
    signed: true
    symmetric: true
    granularity: per_tensor
    scale_constraint: power_of_two

  bias:
    bits: 8
    signed: true
    scale_rule: input_scale_times_weight_scale

  accumulator:
    bits: 32

  requantization:
    power_of_two_only: true

  preprocessing:
    image_divisor: 256
```

The quantizer must derive behavior from configuration rather than hard-coded assumptions.

---

# 32. Hardware Compatibility Comes Before Algorithm Convenience

Before implementing an algorithm, check whether it requires unsupported features.

Do not silently introduce:

```text
arbitrary floating-point scale
per-channel activation quantization
floating zero-point
non-zero zero-point
mixed bit width
learned arbitrary scale
floating-point requantization
unsupported bias precision
```

unless explicitly enabled by the target hardware profile.

A numerically better result that cannot run on target hardware is not a valid improvement.

---

# 33. Required Unit Tests

At minimum test:

## Po2 scale

```text
log2(scale)
```

must be integer within tolerance.

## Quantization

Known values should produce expected integer outputs.

## Dequantization

Known integer values should reconstruct correctly.

## Saturation

Test values outside each supported range.

## Rounding

Test half-way boundaries.

## `/256`

Verify:

```text
0   -> 0.0
128 -> 0.5
255 -> 0.99609375
```

## Bias

Verify:

```text
Sb = Sx * Sw
```

and:

```text
qb = round(b / Sb)
```

for:

```text
INT8
INT16
INT32
```

including saturation.

## Requantization

Verify correct Po2 shift between accumulator and output scales.

## Calibration consistency

Ensure calibration and deployment preprocessing are identical by default.

---

# 34. Numerical Equivalence Tests

Whenever preprocessing or graph transformations are performed, first compare in floating point.

For example:

```text
Original FP32 Model
vs
FP32 Model With Folded Preprocessing
```

The outputs should be numerically equivalent within expected tolerance before quantization is introduced.

Do not debug quantization and graph transformation errors simultaneously.

---

# 35. Reproducibility

Every experiment should log:

```text
git commit
model path/hash
dataset
calibration sample IDs
sample count
preprocessing
weight bits
activation bits
bias bits
accumulator bits
signed ranges
rounding mode
saturation mode
Po2 candidate range
optimization objective
selected exponents
task metric
```

Results without reproducible configuration are not considered reliable.

---

# 36. Agent Rules

When working on this repository:

1. Inspect existing implementation before modifying quantization behavior.
2. Trace the complete preprocessing path.
3. Determine actual integer ranges.
4. Determine actual rounding behavior.
5. Determine accumulator precision.
6. Determine bias precision and bias scale convention.
7. Determine requantization behavior.
8. Verify `/255` vs `/256`.
9. Preserve calibration/deployment consistency.
10. Prefer small, reviewable changes.
11. Add tests for every quantization semantic change.
12. Avoid unrelated refactoring.
13. Never assume TensorFlow/TFLite behavior equals target hardware behavior.
14. Never silently relax hardware constraints to improve accuracy.

When behavior is unclear, inspect code or hardware documentation instead of guessing.

---

# 37. Research Philosophy

The project should progress from simple baselines toward increasingly realistic optimization.

Recommended progression:

```text
MinMax Po2
    ↓
Tensor MSE Po2
    ↓
Clipping-aware Po2
    ↓
Layer reconstruction
    ↓
Joint Weight + Activation
    ↓
Joint Weight + Activation + Bias
    ↓
Full integer datapath optimization
    ↓
Sensitivity / second-order optimization
    ↓
Cross-layer / model-level optimization
```

At every stage:

```text
measure task accuracy
compare against previous baseline
verify hardware compatibility
```

Do not skip directly to complex algorithms without establishing strong baselines.

---

# 38. Main Optimization Objective

The eventual goal is not merely:

```text
min ||X - Xq||
```

or:

```text
min ||W - Wq||
```

The preferred direction is:

```text
min Error(
    ModelFP32(X),
    ModelIntegerPo2(X)
)
```

subject to:

```text
Weight      = INT8
Activation  = INT8
Scale       = 2^k
Bias        = target-specific INT8/16/32
Accumulator = target-specific
Rounding    = hardware-specific
Saturation  = hardware-specific
Preprocess  = deployment-identical
```

For tractability, optimization may be performed layer-by-layer or block-by-block.

---

# 39. Core Principles

Keep these rules in mind at all times.

### Rule 1

> Calibrate the model that will actually run.

### Rule 2

> Calibration must use the input distribution that the hardware will actually see.

### Rule 3

> Po2 exponent selection is a discrete optimization problem.

### Rule 4

> Zero clipping is not automatically optimal.

### Rule 5

> Bias precision is an independent hardware constraint.

### Rule 6

> INT8-bias optimum is not automatically the INT32-bias optimum.

### Rule 7

> Optimize the real integer datapath, not only fake-quant tensor reconstruction.

### Rule 8

> Hardware compatibility is more important than algorithmic convenience.

### Rule 9

> End-to-end task accuracy is the final metric.

---

# 40. Current Default Assumptions

Until target-specific configuration says otherwise:

```text
Weight:
    INT8
    symmetric
    per-tensor
    Po2 scale

Activation:
    INT8
    symmetric
    per-tensor
    Po2 scale

Bias:
    configurable INT8 / INT16 / INT32
    scale = input_scale * weight_scale

Accumulator:
    wide integer accumulator
    default assumption INT32

Input preprocessing:
    image / 256

Calibration preprocessing:
    identical to deployment

Scale optimization:
    discrete exponent search

Primary objective:
    end-to-end task accuracy

Secondary objective:
    layer-output reconstruction

Proxy objective:
    tensor reconstruction error
```

These are defaults, not universal truths.

If actual hardware behavior differs, update the hardware profile rather than silently modifying the quantization algorithm.
