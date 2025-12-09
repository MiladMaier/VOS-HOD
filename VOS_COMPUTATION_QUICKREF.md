# VOS Score Computation: Quick Reference

## What You Need to Know

If you're using this repository for OOD detection with VOS, be aware that there are **two different implementations** of the VOS energy score computation:

### 1. Original VOS Implementation (Correct) ✅

**Used in:**
- Training: `detection/modeling/roihead_gmm.py`
- Inference: `detection/inference/rcnn_predictor.py`
- Running: `python apply_net.py` (full inference pipeline)

**Formula:** `energy = log(sum(w_i * exp(logit_i)))`

### 2. Simplified Implementation (Different) ⚠️

**Used in:**
- Post-processing: `detection/add_vos_scores.py`

**Formula:** `energy = log(sum(exp(w_i * logit_i)))`

## Which Should You Use?

### For Exact Reproduction of Results:
```bash
# Run full inference with the model (recommended)
python apply_net.py --config-file VOC-Detection/faster-rcnn/vos.yaml \
                    --inference-config Inference/standard_nms.yaml \
                    --test-dataset voc_custom_val \
                    --image-corruption-level 0 \
                    --random-seed 0
```

### For Quick Post-Processing (Approximate):
```bash
# Add scores to existing pickle files (faster but slightly different results)
python add_vos_scores.py
```

## Understanding the Difference

The two methods produce **different energy scores**:

| Scenario | Difference |
|----------|-----------|
| Uniform weights (≈1.0) | Negligible (<0.1%) |
| Realistic weights (0.5-2.0) | Moderate (4-20%) |
| Extreme weights | Large (up to 48%) |

**Example:** With weights [0.5, 1.5, 0.8, 1.2] and typical logits:
- Original VOS energy: 3.95
- add_vos_scores energy: 4.75
- **Difference: 20.3%**

## Want to Learn More?

See these files for detailed information:
- **VOS_COMPUTATION_DIFFERENCES.md** - Complete mathematical explanation
- **demonstrate_vos_difference.py** - Run concrete numerical examples

## Quick Demo

```bash
# See the numerical difference yourself
python demonstrate_vos_difference.py
```

This will show side-by-side comparisons with multiple examples.

## Bottom Line

- ✅ Use `apply_net.py` for paper reproduction and benchmarking
- ⚠️ Use `add_vos_scores.py` only for quick experiments (results may vary)
- 📚 Read VOS_COMPUTATION_DIFFERENCES.md for full details
