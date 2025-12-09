"""
Demonstration script showing the numerical difference between 
original VOS energy computation and add_vos_scores.py computation

This script provides concrete examples of how the two methods diverge.
"""

import torch
import torch.nn.functional as F

def original_vos_energy(logits, weights):
    """
    Original VOS implementation: log(sum(w_i * exp(logit_i)))
    
    This uses numerically stable computation:
    energy = m + log(sum(w_i * exp(logit_i - m)))
    """
    m, _ = torch.max(logits, dim=1, keepdim=True)
    value0 = logits - m
    weights_relu = F.relu(weights)
    energy = m + torch.log(torch.sum(
        weights_relu * torch.exp(value0), dim=1, keepdim=True))
    return energy

def add_vos_scores_energy(logits, weights):
    """
    add_vos_scores.py implementation: log(sum(exp(w_i * logit_i)))
    
    This applies weights before exponentiation
    """
    weights_relu = F.relu(weights)
    weighted_logits = logits * weights_relu
    energy = torch.logsumexp(weighted_logits, dim=1, keepdim=True)
    return energy

def demonstrate_difference():
    """Show concrete examples of how the two methods differ"""
    
    print("=" * 80)
    print("VOS Energy Score Computation: Numerical Comparison")
    print("=" * 80)
    
    # Example 1: Uniform weights (close to 1)
    print("\n" + "=" * 80)
    print("Example 1: Uniform weights close to 1.0")
    print("=" * 80)
    logits1 = torch.tensor([[2.0, 3.0, 1.5, 2.5]])
    weights1 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    
    original = original_vos_energy(logits1, weights1)
    modified = add_vos_scores_energy(logits1, weights1)
    
    print(f"Logits:                  {logits1[0].tolist()}")
    print(f"Weights:                 {weights1[0].tolist()}")
    print(f"Original VOS energy:     {original.item():.6f}")
    print(f"add_vos_scores energy:   {modified.item():.6f}")
    print(f"Absolute difference:     {abs(original.item() - modified.item()):.6f}")
    print(f"Relative difference:     {abs(original.item() - modified.item()) / original.item() * 100:.3f}%")
    
    # Example 2: Varying weights
    print("\n" + "=" * 80)
    print("Example 2: Varying weights (realistic scenario)")
    print("=" * 80)
    logits2 = torch.tensor([[2.0, 3.0, 1.5, 2.5]])
    weights2 = torch.tensor([[0.5, 1.5, 0.8, 1.2]])
    
    original = original_vos_energy(logits2, weights2)
    modified = add_vos_scores_energy(logits2, weights2)
    
    print(f"Logits:                  {logits2[0].tolist()}")
    print(f"Weights:                 {weights2[0].tolist()}")
    print(f"Original VOS energy:     {original.item():.6f}")
    print(f"add_vos_scores energy:   {modified.item():.6f}")
    print(f"Absolute difference:     {abs(original.item() - modified.item()):.6f}")
    print(f"Relative difference:     {abs(original.item() - modified.item()) / original.item() * 100:.3f}%")
    
    # Example 3: Extreme weights
    print("\n" + "=" * 80)
    print("Example 3: Extreme weight variation")
    print("=" * 80)
    logits3 = torch.tensor([[2.0, 3.0, 1.5, 2.5]])
    weights3 = torch.tensor([[0.1, 2.0, 0.5, 1.8]])
    
    original = original_vos_energy(logits3, weights3)
    modified = add_vos_scores_energy(logits3, weights3)
    
    print(f"Logits:                  {logits3[0].tolist()}")
    print(f"Weights:                 {weights3[0].tolist()}")
    print(f"Original VOS energy:     {original.item():.6f}")
    print(f"add_vos_scores energy:   {modified.item():.6f}")
    print(f"Absolute difference:     {abs(original.item() - modified.item()):.6f}")
    print(f"Relative difference:     {abs(original.item() - modified.item()) / original.item() * 100:.3f}%")
    
    # Example 4: Batch demonstration
    print("\n" + "=" * 80)
    print("Example 4: Batch processing (5 detections)")
    print("=" * 80)
    torch.manual_seed(42)
    logits_batch = torch.randn(5, 20)  # 5 detections, 20 classes (like VOC)
    weights_batch = torch.rand(1, 20) * 2  # Random weights between 0 and 2
    
    original_batch = original_vos_energy(logits_batch, weights_batch)
    modified_batch = add_vos_scores_energy(logits_batch, weights_batch)
    
    print(f"Number of detections: 5")
    print(f"Number of classes: 20")
    print(f"\nOriginal VOS energies:")
    for i, energy in enumerate(original_batch):
        print(f"  Detection {i+1}: {energy.item():.6f}")
    
    print(f"\nadd_vos_scores energies:")
    for i, energy in enumerate(modified_batch):
        print(f"  Detection {i+1}: {energy.item():.6f}")
    
    print(f"\nPer-detection differences:")
    for i in range(5):
        diff = abs(original_batch[i].item() - modified_batch[i].item())
        rel_diff = diff / original_batch[i].item() * 100
        print(f"  Detection {i+1}: {diff:.6f} ({rel_diff:.3f}%)")
    
    mean_diff = torch.mean(torch.abs(original_batch - modified_batch)).item()
    mean_rel_diff = mean_diff / torch.mean(original_batch).item() * 100
    print(f"\nMean absolute difference: {mean_diff:.6f} ({mean_rel_diff:.3f}%)")
    
    # Mathematical explanation
    print("\n" + "=" * 80)
    print("Mathematical Explanation")
    print("=" * 80)
    print("""
The two formulations are fundamentally different:

Original VOS:     energy = log(sum(w_i * exp(logit_i)))
add_vos_scores:   energy = log(sum(exp(w_i * logit_i)))

Key insight: w * exp(x) ≠ exp(w * x)

When w = 2, x = 1:
  - Original:     2 * exp(1) = 2 * 2.718 = 5.436
  - Modified:     exp(2 * 1) = exp(2) = 7.389
  - Difference:   36% higher in modified version

This difference propagates through the logistic regression, potentially
affecting OOD detection performance metrics (AUROC, FPR95, AUPR).
    """)
    
    print("=" * 80)
    print("Conclusion")
    print("=" * 80)
    print("""
For faithful reproduction of the original VOS method:
1. Use the log_sum_exp method from roihead_gmm.py
2. Or run full inference with apply_net.py instead of post-processing

The add_vos_scores.py script is a convenient post-processing tool but
may produce slightly different results from the full inference pipeline.
    """)

if __name__ == "__main__":
    demonstrate_difference()
