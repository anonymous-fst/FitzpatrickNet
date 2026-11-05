"""
fst — shared internal package for Fitzpatrick Skin Type classification.

Submodules
----------
config          DATA_SPLIT_VARIANTS registry and CSV path helpers
reproducibility set_seed
soft_labels     Triangular / adaptive soft-label generation and alpha annealing
losses          FocalKLLoss, class weights, patient consistency, contrastive loss
sampling        BalancedBatchSampler, mixup_batch, provenance weight helpers
metrics         Extended metric suite, bootstrap CIs, patient-level metrics
uncertainty     MC-Dropout prediction, ECE helpers, uncertainty box-plot
plotting        All visualisation utilities (CM, ROC, PR, calibration, bars …)
explainability  MinimalGradCAM and Grad-CAM save helpers
"""
