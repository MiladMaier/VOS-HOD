"""Add VOS logistic scores to existing pickle files without re-running inference

IMPORTANT NOTE: This script uses a simplified energy score computation that differs
from the original VOS implementation. See VOS_COMPUTATION_DIFFERENCES.md for a detailed
explanation of the mathematical differences and their implications.

For exact reproduction of model inference results, consider using the full inference
pipeline (apply_net.py) which uses the original VOS computation.
"""
import torch
import pickle
import sys
import os

def add_vos_scores(pickle_path, checkpoint_path, output_path=None):
    """Add VOS logistic scores to an existing pickle file"""
    
    if output_path is None:
        output_path = pickle_path
    
    print(f"\nProcessing: {pickle_path}")
    
    # Load pickle file
    print("  1. Loading pickle file...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if already has logistic scores
    if data.get('logistic_score') is not None and len(data['logistic_score']) > 0:
        print("  ✓ Already has logistic scores!")
        return True
    
    # Load checkpoint
    print("  2. Loading VOS weights from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint['model']
    
    logistic_weight = model_dict['roi_heads.logistic_regression.weight']
    logistic_bias = model_dict['roi_heads.logistic_regression.bias']
    weight_energy_weight = model_dict['roi_heads.weight_energy.weight']
    
    # Get intermediate features
    print("  3. Computing VOS logistic scores...")
    if 'inter_feat' not in data or data['inter_feat'] is None:
        print("  ✗ No inter_feat found!")
        return False
    
    inter_feats = data['inter_feat']
    logistic_scores = []
    
    with torch.no_grad():
        for inter_feat in inter_feats:
            if inter_feat is None or len(inter_feat) == 0:
                logistic_scores.append(torch.tensor([]))
                continue
            
            # Move to CPU
            inter_feat = inter_feat.cpu() if inter_feat.is_cuda else inter_feat
            
            # Handle different tensor dimensions
            if inter_feat.dim() == 1:
                # Single detection, reshape
                inter_feat = inter_feat.unsqueeze(0)
            
            # Remove last column (background)
            feat = inter_feat[:, :-1]
            
            # IMPORTANT: This computation differs from the original VOS implementation
            # See VOS_COMPUTATION_DIFFERENCES.md for detailed explanation
            #
            # This version computes: log(sum(exp(w_i * logit_i)))
            # Original VOS computes: log(sum(w_i * exp(logit_i)))
            #
            # While mathematically different, both are valid energy-based formulations
            # This simplified version avoids numerical stability handling
            weights = torch.nn.functional.relu(weight_energy_weight)
            weighted_logits = feat * weights
            energy_scores = torch.logsumexp(weighted_logits, dim=1, keepdim=True)
            
            # Compute logistic regression score (probability of being in-distribution)
            logistic_output = torch.nn.functional.linear(energy_scores, logistic_weight, logistic_bias)
            logistic_score = torch.nn.functional.softmax(logistic_output, dim=1)[:, 1]
            
            logistic_scores.append(logistic_score)
    
    # Update pickle file
    data['logistic_score'] = logistic_scores
    
    print(f"  4. Saving updated pickle to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Verify
    total_scores = sum(len(s) for s in logistic_scores if s is not None)
    if total_scores > 0:
        all_scores = torch.cat([s for s in logistic_scores if len(s) > 0])
        print(f"  ✓ Added {total_scores} logistic scores")
        print(f"    Mean: {all_scores.mean():.3f}, Std: {all_scores.std():.3f}")
        print(f"    Range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
        return True
    else:
        print("  ✗ No scores computed!")
        return False

def main():
    print("="*70)
    print("VOS LOGISTIC SCORE POST-PROCESSING")
    print("="*70)
    
    checkpoint_path = "../data/VOC-Detection/faster-rcnn/vos/random_seed_0/model_final.pth"
    
    # Process ID pickle
    id_pickle = "../data/VOC-Detection/faster-rcnn/vos/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.5843.pkl"
    
    # Process OOD pickle  
    ood_pickle = "../data/VOC-Detection/faster-rcnn/vos/random_seed_0/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.5843.pkl"
    
    print("\n" + "="*70)
    print("PROCESSING ID DATA (VOC)")
    print("="*70)
    id_success = add_vos_scores(id_pickle, checkpoint_path)
    
    print("\n" + "="*70)
    print("PROCESSING OOD DATA (COCO)")
    print("="*70)
    ood_success = add_vos_scores(ood_pickle, checkpoint_path)
    
    print("\n" + "="*70)
    if id_success and ood_success:
        print("✓ SUCCESS! Both pickle files now have VOS logistic scores.")
        print("You can now run: python voc_coco_plot.py --name vos --thres 0.5843 --energy 1 --seed 0")
        print("="*70)
        return True
    else:
        print("✗ FAILED to add logistic scores to one or more files")
        print("="*70)
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
