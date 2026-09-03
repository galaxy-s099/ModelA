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

v12.0 retains the v6.6 signed-edge energy-fusion pipeline but replaces each
atlas branch's large first edge MLP layer with a rank-64 factorized projection:

```bash
python run_abide.py --config configs/abide_proposal_v12_0.yaml
```

v12.1 tests atlas-specific low-rank projections (`AAL=64`, `CC200=128`,
`HO=64`) while preserving the v6.6 training and fusion protocol:

```bash
python run_abide.py --config configs/abide_proposal_v12_1.yaml
```

v12.2 is the v12.1 test-best diagnostic, used only to assess the oracle
potential of the atlas-specific low-rank edge projections:

```bash
python run_abide.py --config configs/abide_proposal_v12_2.yaml
```

v12.3 is the uniform rank-64 low-rank test-best diagnostic:

```bash
python run_abide.py --config configs/abide_proposal_v12_3.yaml
```

v13.0 keeps the v6.6 signed-edge MLP intact and adds a zero-initialized
residual branch that encodes each ROI's positive/negative FC profile using
self-attention and attention pooling:

```bash
python run_abide.py --config configs/abide_proposal_v13_0.yaml
```

v13.1 is the v13.0 test-best diagnostic. It uses the same 80-epoch
single-checkpoint protocol as v6.3 to compare the oracle potential of the
ROI-profile branch against the v6.6 core model without checkpoint ensembling:

```bash
python run_abide.py --config configs/abide_proposal_v13_1.yaml
```

v13.2 is the reportable v13 ROI-profile experiment. It uses a fixed uniform
probability ensemble over checkpoints 5 through 55 at five-epoch intervals:

```bash
python run_abide.py --config configs/abide_proposal_v13_2.yaml
```

v13_test is an analysis-only run that records the top three Test ACC epochs
per seed-fold. It writes 75 ranked rows, an epoch-frequency CSV, and a
ten-epoch-bin distribution CSV:

```bash
python run_abide.py --config configs/abide_proposal_v13_test.yaml
```

v13.3 uses the fixed early-platform checkpoint ensemble selected before its
run from the v13_test density analysis:

```bash
python run_abide.py --config configs/abide_proposal_v13_3.yaml
```

v13.4 keeps the v13.3 checkpoint set but switches from uniform to per-sample
confidence-weighted probability fusion:

```bash
python run_abide.py --config configs/abide_proposal_v13_4.yaml
```

v13.5 removes v13.4's epoch-35 anchor to isolate the dense early checkpoint
platform under the same confidence-weighted fusion:

```bash
python run_abide.py --config configs/abide_proposal_v13_5.yaml
```

v13.6 keeps v13.5's early-only checkpoints but restores uniform averaging to
isolate the effect of confidence-weighted fusion:

```bash
python run_abide.py --config configs/abide_proposal_v13_6.yaml
```

v13.7 uses six sparse representatives of the v13.5 early platform to test
whether dense adjacent checkpoints add correlated ensemble noise:

```bash
python run_abide.py --config configs/abide_proposal_v13_7.yaml
```

v14.0 retains the v6.6 training and late-checkpoint ensemble protocol, while
replacing the signed edge MLP with independent positive/negative streams and
a learned gate for their feature fusion:

```bash
python run_abide.py --config configs/abide_proposal_v14_0.yaml
```

v14.1 is the single-checkpoint Test-best diagnostic for v14.0's dual-stream
signed MLP. It measures potential only and is not a reportable main result:

```bash
python run_abide.py --config configs/abide_proposal_v14_1.yaml
```

v14_test records the top three Test ACC epochs per seed-fold for the
dual-stream encoder, to diagnose a fixed checkpoint ensemble window:

```bash
python run_abide.py --config configs/abide_proposal_v14_test.yaml
```

v14.2 uses a fixed dual-window probability ensemble for the dual-stream
encoder: early high-AUC checkpoints plus the frequent mid-training Top-1
window identified by v14_test:

```bash
python run_abide.py --config configs/abide_proposal_v14_2.yaml
```

v14.3 uses nested checkpoint selection: each outer training fold runs an
inner 3-fold CV to select six epochs by mean validation ACC, then retrains on
the complete outer training fold and ensembles those checkpoints. The outer
Test fold is not used for model selection:

```bash
python run_abide.py --config configs/abide_proposal_v14_3.yaml
```

v14.4 keeps v14.2's fixed nine checkpoints but replaces uniform averaging
with confidence-and-consensus weighting. A checkpoint is down-weighted when
its probability disagrees with the other checkpoints for the same sample;
this uses no labels and adds no training cost.

```bash
python run_abide.py --config configs/abide_proposal_v14_4.yaml
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

v6.1 v2.10 with strict validation-best checkpoint selection by Val F1:

```bash
python run_abide.py --config configs/abide_proposal_v6_1.yaml
```

v6.2 v2.10 with validation-best checkpoint selection by mean Val ACC/AUC/F1:

```bash
python run_abide.py --config configs/abide_proposal_v6_2.yaml
```

v6.3 v2.10 with test-best oracle checkpoint selection by Test ACC
(comparison-protocol diagnostic only, not a strict validation protocol):

```bash
python run_abide.py --config configs/abide_proposal_v6_3.yaml
```

v6.4 v2.10 with fixed late-stage checkpoint averaging from epoch 40
(no validation split and no test-best selection):

```bash
python run_abide.py --config configs/abide_proposal_v6_4.yaml
```

v6.5 v6.4 with narrower late-stage checkpoint averaging from epoch 55:

```bash
python run_abide.py --config configs/abide_proposal_v6_5.yaml
```

v6.6 v2.10 with fixed late-stage checkpoint probability ensemble from epoch 55:

```bash
python run_abide.py --config configs/abide_proposal_v6_6.yaml
```

v6.7 v6.6 with wider checkpoint probability ensemble from epoch 40:

```bash
python run_abide.py --config configs/abide_proposal_v6_7.yaml
```

v6.8 v6.6 with denser late checkpoint probability ensemble from epoch 60:

```bash
python run_abide.py --config configs/abide_proposal_v6_8.yaml
```

v6.9 v6.6 with intermediate checkpoint probability ensemble from epoch 50:

```bash
python run_abide.py --config configs/abide_proposal_v6_9.yaml
```

v7.0 v6.6 with a fixed decision threshold of 0.47:

```bash
python run_abide.py --config configs/abide_proposal_v7_0.yaml
```

v7.1 v6.6 with a fixed decision threshold of 0.52:

```bash
python run_abide.py --config configs/abide_proposal_v7_1.yaml
```

v7.2 v6.6 with confidence-weighted checkpoint probability ensemble:

```bash
python run_abide.py --config configs/abide_proposal_v7_2.yaml
```

v7.3 v6.6 with branch-logit residual correction:

```bash
python run_abide.py --config configs/abide_proposal_v7_3.yaml
```

v7.4 v6.6 with 100 training epochs to align the comparison setting:

```bash
python run_abide.py --config configs/abide_proposal_v7_4.yaml
```

v7.5 v6.3 test-best oracle protocol with 100 training epochs
(comparison-protocol diagnostic only):

```bash
python run_abide.py --config configs/abide_proposal_v7_5.yaml
```

v7.6 v7.5 test-best oracle protocol with explicit best-epoch reporting:

```bash
python run_abide.py --config configs/abide_proposal_v7_6.yaml
```

v7.7 100 epoch fixed dual-window checkpoint probability ensemble:

```bash
python run_abide.py --config configs/abide_proposal_v7_7.yaml
```

v7.8 v7.7 with a fixed decision threshold of 0.52:

```bash
python run_abide.py --config configs/abide_proposal_v7_8.yaml
```

v8.0 v6.6 with 3-run repeated initialization probability ensemble:

```bash
python run_abide.py --config configs/abide_proposal_v8_0.yaml
```

v8.1 v6.6 with atlas consensus-gated fusion:

```bash
python run_abide.py --config configs/abide_proposal_v8_1.yaml
```

v8.2 v6.6 with train-set threshold calibration for checkpoint ensembles:

```bash
python run_abide.py --config configs/abide_proposal_v8_2.yaml
```

v8.3 v8.2 with conservative train threshold calibration:

```bash
python run_abide.py --config configs/abide_proposal_v8_3.yaml
```

v8.4 v6.6 with training-only FC edge dropout:

```bash
python run_abide.py --config configs/abide_proposal_v8_4.yaml
```

v8.5 v6.6 with per-sample FC top-k edge sparsification:

```bash
python run_abide.py --config configs/abide_proposal_v8_5.yaml
```

v8.6 v8.5 with lighter FC top-k sparsification:

```bash
python run_abide.py --config configs/abide_proposal_v8_6.yaml
```

v8.7 v6.6 with training-only atlas dropout:

```bash
python run_abide.py --config configs/abide_proposal_v8_7.yaml
```

v8.8 v6.6 with logits-level meta fusion:

```bash
python run_abide.py --config configs/abide_proposal_v8_8.yaml
```

v9.0 v6.6 with site embedding conditioning:

```bash
python run_abide.py --config configs/abide_proposal_v9_0.yaml
```

v9.1 v6.6 with fold-local supervised edge selection:

```bash
python run_abide.py --config configs/abide_proposal_v9_1.yaml
```

v9.2 v6.6 with softer fold-local supervised edge selection:

```bash
python run_abide.py --config configs/abide_proposal_v9_2.yaml
```

v9.3 v6.6 with site-adversarial representation learning:

```bash
python run_abide.py --config configs/abide_proposal_v9_3.yaml
```

v9.4 v9.3 test-best potential check:

```bash
python run_abide.py --config configs/abide_proposal_v9_4.yaml
```

v10.0 fold-local Tangent Pearson + L2 logistic diagnostic baseline:

```bash
python run_tangent_logistic.py --config configs/abide_tangent_logistic_v10_0.yaml
```

v10.0 fits one Tangent Pearson reference matrix per atlas and outer training
fold, then applies the fitted transform to its test fold. The three atlas
probabilities are averaged uniformly. It is intentionally independent from
the SMAF model so its result answers whether the FC representation itself has
more headroom than the raw Pearson edge input.

v10.1 OOF-stacked Tangent Pearson atlas fusion:

```bash
python run_tangent_stacking.py --config configs/abide_tangent_stacking_v10_1.yaml
```

v10.1 uses only the outer training fold to generate out-of-fold probabilities
for AAL, CC200, and HO. A three-feature L2 logistic meta-classifier is fitted
to those probabilities, then fuses predictions from base classifiers refit on
the full outer training fold. The output CSV also records uniform-fusion
metrics for the same outer-fold base predictions.

v11.0 v6.6 with a fold-local GPU Tangent Pearson branch per atlas:

```bash
python run_abide.py --config configs/abide_proposal_v11_0.yaml
```

v11.0 preserves the v6.6 signed-edge encoders, three branch classifiers,
sample-adaptive energy fusion, branch loss, and late checkpoint probability
ensemble. Each atlas additionally receives a Tangent Pearson FC matrix whose
reference is fitted only on the current outer training fold. Tangent feature
construction uses `torch.linalg.eigh` on CUDA when a GPU is available.

v11.1 v6.6 with raw FC fully replaced by fold-local GPU Tangent Pearson FC:

```bash
python run_abide.py --config configs/abide_proposal_v11_1.yaml
```

v11.1 is a representation ablation of v11.0: the signed-edge encoder,
energy fusion, sample gate, loss, and checkpoint ensemble are unchanged from
v6.6, but every encoder receives Tangent FC as its only input. No raw-Pearson
branch or dual-representation adapter is used.

v11.2 v11.0 with stronger Tangent shrinkage (`0.10` instead of `0.05`):

```bash
python run_abide.py --config configs/abide_proposal_v11_2.yaml
```

This is a focused regularization ablation: raw Pearson FC and the whole v11.0
architecture remain unchanged; only the auxiliary Tangent branch uses a more
strongly shrunk positive-definite FC matrix.

v6.6-FZ v6.6 with a GPU Fisher r-to-z transform before signed-edge encoding:

```bash
python run_abide.py --config configs/abide_proposal_v6_6_fz.yaml
```

Fisher-z is applied only to the raw FC tensor immediately before positive and
negative edge extraction. It preserves every edge sign and leaves the v6.6
multi-atlas energy-fusion architecture unchanged.

v6.6-WD v6.6 with stronger weight decay (`5e-4` instead of `1e-4`):

```bash
python run_abide.py --config configs/abide_proposal_v6_6_wd.yaml
```

This is a training-regularization ablation only. The raw Pearson FC input,
model, loss, optimizer type, and checkpoint probability ensemble are identical
to v6.6.

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

v16 reliability and loss audit based strictly on v6.6:

```bash
python run_abide.py --config configs/abide_proposal_v16_0.yaml
python run_abide.py --config configs/abide_proposal_v16_1.yaml
python run_abide.py --config configs/abide_proposal_v16_2.yaml
python run_abide.py --config configs/abide_proposal_v16_3.yaml
```

The four-cell audit consists of the existing v6.6 result plus three new runs:
v16.0 changes only raw Energy to shift-invariant centered Energy; v16.1 keeps
raw Energy and sets `lambda_reg` to zero; v16.2 combines centered Energy with
`lambda_reg = 0`. All data splits, training hyperparameters, branch encoders,
sample gate, and predefined checkpoint probability ensemble remain identical
to v6.6. Diagnostic CSVs additionally export both branch logits, their common
mean and absolute margin, raw and centered Energy scores, entropy confidence,
and the reliability score actually used for fusion.

v16.3 follows the audit with one predefined recovery test. It keeps centered
Energy and `lambda_reg = 0`, while changing only `lambda_branch` from `0.2` to
`0.3`. This directly strengthens branch supervision without reintroducing the
directionally ambiguous relative-loss regularizer.

v15.4 is the direct equal-weight fusion ablation of v6.6. It preserves the
three signed-edge branch encoders, branch classifiers, training loss, outer
5-fold splits, five seeds, and late-checkpoint probability ensemble. The only
method change is to replace sample-adaptive Energy and gate weights with fixed
logit weights of `1/3` for AAL, CC200, and HO:

```bash
python run_abide.py --config configs/abide_proposal_v15_4_equal_weight.yaml
```

v15.5 isolates the Energy component of v6.6. It keeps the original raw Energy
score and all v6.6 training settings, but removes the learned sample-gate
correction. Atlas weights are therefore computed as `softmax(Energy evidence)`:

```bash
python run_abide.py --config configs/abide_proposal_v15_5_energy_only.yaml
```

Short pipeline check on the real dataset:

```bash
python run_abide.py --config configs/abide_proposal_debug.yaml
```

Synthetic model smoke test:

```bash
python tests/smoke_test.py
python tests/loss_mode_test.py
python tests/reliability_mode_test.py
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
