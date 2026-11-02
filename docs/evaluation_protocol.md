# Evaluation protocol

The benchmark uses UCI Iranian Churn (3,150 rows, 495 positive labels). Raw fields are validated and `Status` is omitted due unclear time availability and strong outcome association. Group IDs are stable hashes of the 12 model input fields; profiles with identical predictor values—including rows whose labels disagree—stay together.

A shuffled, stratified 5-fold group split with seed 2026 provides a single held-out fold. The remaining development rows use 5-fold stratified group cross-validation (seed 2027). All preprocessing and feature transformations are inside each fold's scikit-learn pipeline. The model is selected by the highest mean inner-fold average precision. Each candidate's classification threshold is chosen by maximum F1 on its development-only out-of-fold probabilities, then the final candidate is fit on all development rows and evaluated once on the reserved fold.

The test fold is not used to choose the model, threshold, or features. Held-out permutation importance is included only as post-evaluation description; it does not feed training or selection. Thresholded metrics are specific to this split and the chosen F1 objective. The evaluation is not temporal, calibrated, causal, or externally validated.
