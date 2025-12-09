We express our gratitude to Xuefeng Du, Zhaoning Wang, Mu Cai and Yixuan Li for creating Virtual Outlier Synthesis. Original repository of VOS: https://github.com/deeplearning-wisc/vos.
Our repository extends VOS with higher-order discrimination (VOS-HOD) to improve out-of-distribution detection performance in object detection.

## ⚠️ Important Note: VOS Score Computation

This repository contains **two different implementations** of VOS energy score computation. For accurate reproduction of results:

- ✅ **Recommended**: Use the full inference pipeline (`apply_net.py`) which implements the original VOS method correctly
- ⚠️ **Alternative**: The post-processing script (`add_vos_scores.py`) uses a simplified computation that may produce different results

**For details, see:**
- [Quick Reference Guide](VOS_COMPUTATION_QUICKREF.md) - Start here
- [Detailed Technical Explanation](VOS_COMPUTATION_DIFFERENCES.md) - Mathematical analysis
- [Numerical Demonstration](demonstrate_vos_difference.py) - Run concrete examples

The difference can be significant (4-48%) depending on the learned weight values. For benchmarking and paper reproduction, always use `apply_net.py`.
