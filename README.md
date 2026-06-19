# SMAF Proposal Net

Strict implementation of the proposed signed multi-atlas brain functional
network framework for ABIDE ASD vs TC classification.

This project is built independently in `D:\Model`. The earlier
`D:\SMAF-Net` project remains unchanged and can be used as the baseline
comparison.

## Architecture

For each atlas, the model receives a functional connectivity matrix and splits
it into positive and negative adjacency matrices:

```text
A+ = max(FC, 0)
A- = max(-FC, 0)
```

Each atlas branch follows the proposal's two-state signed propagation:

```text
H_pos^(1) = activation(A+ X W_pos)
H_neg^(1) = activation(A- X W_neg)

H_pos^(l) = activation(concat(A+ H_pos^(l-1) W_pp,
                              A- H_neg^(l-1) W_nn))

H_neg^(l) = activation(concat(A+ H_neg^(l-1) W_pn,
                              A- H_pos^(l-1) W_np))
```

The final positive and negative node states are concatenated and mean-pooled
into an atlas-level embedding. Atlas-level embeddings then attend to the other
atlases only. The attention diagonal is masked so an atlas cannot attend to
itself.

Each enhanced atlas embedding has an independent classifier. Its energy score
defines a decision-fusion weight:

```text
E_i = -T * logsumexp(logits_i / T)
w_i = softmax(-E_i)
fusion_logits = sum(w_i * logits_i)
```

For the v1.2 ablation, the energy path is removed. The three attention-enhanced
atlas embeddings are concatenated and classified by one fusion classifier.

For the v1.1 ablation, cross-atlas attention is also removed. The raw atlas
embeddings from the Signed GCN branches are concatenated and classified.

The training objective follows the proposal:

```text
L = L_fusion + lambda_branch * sum(L_i) + lambda_reg * L_reg
L_reg = sum(max(0, L_i - L_fusion + margin))
```

The original proposal direction remains available as `proposal_literal`.
The `abide_proposal_v1_4.yaml` experiment tests the alternative
`fusion_better` direction:

```text
L_reg = sum(max(0, L_fusion - L_i + margin))
```

## Input Data

The default configuration expects:

```text
labels.npy
X_aal.npy
X_cc200.npy
X_ho.npy
```

Each `X_<atlas>.npy` file has shape `samples x nodes x nodes`.

The default node feature is each subject's FC row. To use separately extracted
ROI/BOLD features, add `node_feature_file` and `feature_dim` to the matching
atlas in the YAML configuration. The feature file must have shape
`samples x nodes x feature_dim`.

## Run

Full repeated cross validation:

```bash
python run_abide.py --config configs/abide_proposal.yaml
```

v1.4 regularization-direction experiment:

```bash
python run_abide.py --config configs/abide_proposal_v1_4.yaml
```

v1.5 deeper signed propagation experiment (`num_signed_layers = 3`):

```bash
python run_abide.py --config configs/abide_proposal_v1_5.yaml
```

v1.6 residual deeper signed propagation experiment
(`num_signed_layers = 3`, `use_signed_residual = true`):

```bash
python run_abide.py --config configs/abide_proposal_v1_6.yaml
```

v1.8 residual signed propagation with learnable signed edge gates
on the rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v1_8.yaml
```

v1.9 deeper residual signed propagation with learnable signed edge gates
on the rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v1_9.yaml
```

v2.0 SMAF-Net v5 signed edge branch encoder with proposal cross-atlas
attention, energy fusion, and loss on the rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v2_0.yaml
```

v2.1 SMAF-Net v5 signed edge branch encoder with proposal cross-atlas
attention and v5-style gated feature fusion on the rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v2_1.yaml
```

v2.2 SMAF-Net v5 signed edge branch encoder with energy fusion and no
cross-atlas attention on the rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v2_2.yaml
```

v2.3 v2.2 loss ablation without confidence regularization:

```bash
python run_abide.py --config configs/abide_proposal_v2_3.yaml
```

v2.4 v2.3 with validation checkpoint and threshold calibration:

```bash
python run_abide.py --config configs/abide_proposal_v2_4.yaml
```

v2.5 v2.2 structure with reduced confidence regularization:

```bash
python run_abide.py --config configs/abide_proposal_v2_5.yaml
```

v2.6 v2.2 structure with intermediate confidence regularization:

```bash
python run_abide.py --config configs/abide_proposal_v2_6.yaml
```

v2.7 v2.2 structure with stronger confidence regularization:

```bash
python run_abide.py --config configs/abide_proposal_v2_7.yaml
```

v2.8 v2.2 structure with weaker branch supervision:

```bash
python run_abide.py --config configs/abide_proposal_v2_8.yaml
```

v2.9 v2.2 structure with learnable atlas prior in energy fusion:

```bash
python run_abide.py --config configs/abide_proposal_v2_9.yaml
```

v2.10 v2.2 structure with sample-adaptive atlas gate in energy fusion:

```bash
python run_abide.py --config configs/abide_proposal_v2_10.yaml
```

v3.0 v2.10 stage baseline with reduced sample gate scale:

```bash
python run_abide.py --config configs/abide_proposal_v3_0.yaml
```

v3.1 v2.10 with zero-initialized residual classifier head:

```bash
python run_abide.py --config configs/abide_proposal_v3_1.yaml
```

v3.2 v2.10 with longer training and lower learning rate:

```bash
python run_abide.py --config configs/abide_proposal_v3_2.yaml
```

v3.3 v2.10 blended with base energy fusion:

```bash
python run_abide.py --config configs/abide_proposal_v3_3.yaml
```

v3.4 v2.10 on the QC10 filtered rebuilt FC dataset:

```bash
python run_abide.py --config configs/abide_proposal_v3_4.yaml
```

v3.5 v2.10 with a light negative-class loss weight for ACC/SPE balance:

```bash
python run_abide.py --config configs/abide_proposal_v3_5.yaml
```

v4.0 v2.10 with a lightweight shared correction head:

```bash
python run_abide.py --config configs/abide_proposal_v4_0.yaml
```

v4.1 v2.10 diagnostic run with per-atlas branch metrics and fusion weights:

```bash
python run_abide.py --config configs/abide_proposal_v4_1.yaml
```

v4.2 v2.10 with a stronger CC200 branch encoder:

```bash
python run_abide.py --config configs/abide_proposal_v4_2.yaml
```

v4.3 v2.10 with sample-wise branch reliability weight alignment:

```bash
python run_abide.py --config configs/abide_proposal_v4_3.yaml
```

v4.4 v4.3 with weaker branch reliability weight alignment:

```bash
python run_abide.py --config configs/abide_proposal_v4_4.yaml
```

v5.0 v2.10 with post-training train-fold threshold calibration:

```bash
python run_abide.py --config configs/abide_proposal_v5_0.yaml
```

v5.1 v2.10 fusion with edge-vector and ROI/node-summary branch encoders:

```bash
python run_abide.py --config configs/abide_proposal_v5_1.yaml
```

v5.2 v2.10 fusion with zero-initialized edge bottleneck residual encoders:

```bash
python run_abide.py --config configs/abide_proposal_v5_2.yaml
```

v6.0 v2.10 with strict validation-best checkpoint selection by Val AUC:

```bash
python run_abide.py --config configs/abide_proposal_v6_0.yaml
```

v1.3 confidence-regularization ablation (`lambda_reg = 0`):

```bash
python run_abide.py --config configs/abide_proposal_v1_3.yaml
```

v1.2 feature-level fusion ablation (no energy decision fusion):

```bash
python run_abide.py --config configs/abide_proposal_v1_2.yaml
```

v1.1 raw atlas embedding concat ablation:

```bash
python run_abide.py --config configs/abide_proposal_v1_1.yaml
```

Short pipeline check on the real dataset:

```bash
python run_abide.py --config configs/abide_proposal_debug.yaml
```

Synthetic model smoke test:

```bash
python tests/smoke_test.py
python tests/loss_mode_test.py
python tests/feature_concat_test.py
python tests/raw_concat_test.py
```

Synthetic end-to-end data and training smoke test:

```bash
python tests/pipeline_smoke_test.py
```

## Discussion Items

The implementation keeps the proposal formula literal where possible. Confirm
these experimental choices before the final ABIDE run:

1. Whether negative adjacency matrices should receive explicit self-loops.
2. Whether FC rows are acceptable node features or separate ROI/BOLD features
   should be exported.
3. Whether `L_reg = sum(max(0, L_i - L_fusion + margin))` has the intended
   optimization direction.
4. Whether site-aware splitting should be added when phenotypic metadata is
   available.
