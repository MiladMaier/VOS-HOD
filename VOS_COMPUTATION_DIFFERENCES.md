# VOS Score Computation: Differences Between add_vos_scores.py and Original Implementation

## Overview

This document explains the computational difference between the VOS (Virtual Outlier Synthesis) score computation in `add_vos_scores.py` and the original VOS implementation used during training and inference in the main model.

## Key Difference: Weighted Energy Score Computation

The primary difference lies in **when and how the learned weights are applied** in the energy score calculation.

### Original VOS Implementation (Training & Model Inference)

**Location:** `detection/modeling/roihead_gmm.py` (lines 729-748), `classification/CIFAR/train_virtual.py` (lines 161-181)

**Method:**
```python
def log_sum_exp(self, value, dim=None, keepdim=False):
    """Numerically stable implementation with learned weights"""
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        # Apply weights to each logit BEFORE summing
        return m + torch.log(torch.sum(
            F.relu(self.weight_energy.weight) * torch.exp(value0),
            dim=dim, keepdim=keepdim))
```

**Computation steps:**
1. Take the maximum value `m` for numerical stability
2. Subtract `m` from all values: `value0 = value - m`
3. **Apply learned weights element-wise**: `F.relu(weight_energy.weight) * exp(value0)`
4. Sum the weighted exponentials
5. Take logarithm and add back `m`

**Mathematical Formula:**
```
energy = m + log(sum(w_i * exp(logit_i - m)))
```
where `w_i = ReLU(weight_energy.weight[i])` are learned per-class weights.

### add_vos_scores.py Implementation (Post-Processing)

**Location:** `detection/add_vos_scores.py` (lines 60-66)

**Method:**
```python
# Compute weighted logsumexp
weights = torch.nn.functional.relu(weight_energy_weight)
weighted_logits = feat * weights
energy_scores = torch.logsumexp(weighted_logits, dim=1, keepdim=True)
```

**Computation steps:**
1. Extract learned weights from checkpoint
2. **Multiply logits by weights element-wise**: `weighted_logits = feat * weights`
3. Apply PyTorch's standard `logsumexp` to weighted logits

**Mathematical Formula:**
```
energy = log(sum(exp(w_i * logit_i)))
```
where `w_i = ReLU(weight_energy.weight[i])` are the same learned weights.

## Mathematical Difference

The two computations are **NOT mathematically equivalent**:

### Original VOS:
```
energy = m + log(sum(w_i * exp(logit_i - m)))
      = log(exp(m) * sum(w_i * exp(logit_i - m)))
      = log(sum(w_i * exp(logit_i)))
```

### add_vos_scores.py:
```
energy = log(sum(exp(w_i * logit_i)))
```

The difference is:
- **Original VOS**: `log(sum(w_i * exp(logit_i)))` — weights are applied to exponentials
- **add_vos_scores.py**: `log(sum(exp(w_i * logit_i)))` — weights are applied to logits before exponentiation

## Impact of the Difference

### When weights are close to 1:
The difference is minimal because:
- Original: `w * exp(x) ≈ exp(x)`
- Modified: `exp(w * x) ≈ exp(x)`

### When weights vary significantly from 1:
The computations diverge because:
```
w * exp(x) ≠ exp(w * x)
```

For example, if `w = 2` and `x = 1`:
- Original: `2 * exp(1) = 2 * 2.718 = 5.436`
- Modified: `exp(2 * 1) = exp(2) = 7.389`

This difference can affect the energy scores and subsequently the logistic regression output.

## Why This Matters

1. **Consistency**: The post-processing script (`add_vos_scores.py`) computes energy scores differently from the model's inference code (`rcnn_predictor.py` line 173)

2. **Reproducibility**: Results from `add_vos_scores.py` may not exactly match results from running full inference with the model

3. **Model Behavior**: The original implementation preserves the mathematical property that energy scores are weighted combinations of class probabilities, while the modified version changes the weighting scheme

## Recommendation

For accurate reproduction of VOS results, use the original `log_sum_exp` implementation:

```python
# Correct implementation matching original VOS
def log_sum_exp_vos(inter_feat, weight_energy_weight):
    """Original VOS energy computation"""
    m, _ = torch.max(inter_feat, dim=1, keepdim=True)
    value0 = inter_feat - m
    weights = torch.nn.functional.relu(weight_energy_weight)
    energy_scores = m + torch.log(torch.sum(
        weights * torch.exp(value0), dim=1, keepdim=True))
    return energy_scores
```

## Files Affected

- **Training/Inference**: `detection/modeling/roihead_gmm.py` (correct implementation)
- **Model Inference**: `detection/inference/rcnn_predictor.py` (correct implementation, uses model's log_sum_exp)
- **Post-Processing**: `detection/add_vos_scores.py` (incorrect implementation, needs fix)

## References

- Original VOS Paper: "VOS: Learning What You Don't Know by Virtual Outlier Synthesis" (Du et al., 2022)
- Original VOS Repository: https://github.com/deeplearning-wisc/vos
