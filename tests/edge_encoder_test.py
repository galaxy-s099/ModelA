import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.edge_encoder import (
    EdgeBranchEncoder,
    ROIProfileAttentionEncoder,
    apply_fc_topk_sparsity,
    fc_to_edge_vector,
    fc_to_node_summary,
)


def main():
    torch.manual_seed(0)
    fc = torch.tensor(
        [
            [
                [0.0, 1.0, -2.0],
                [1.0, 0.0, 3.0],
                [-2.0, 3.0, 0.0],
            ]
        ]
    )
    summary = fc_to_node_summary(fc)
    expected = torch.tensor([[0.5, 2.0, 1.5, 1.0, 0.0, 1.0, 1.5, 2.0, 2.5]])
    assert torch.allclose(summary, expected)
    assert torch.allclose(
        fc_to_edge_vector(fc),
        torch.tensor([[1.0, -2.0, 3.0]]),
    )

    sparse_fc = apply_fc_topk_sparsity(fc, topk_ratio=1.0 / 3.0)
    expected_sparse_fc = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 3.0, 0.0],
            ]
        ]
    )
    assert torch.allclose(sparse_fc, expected_sparse_fc)

    encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        num_nodes=3,
        use_node_summary=True,
        node_summary_hidden_dim=6,
        node_summary_embedding_dim=4,
    )
    encoder.train()
    batch = fc.repeat(4, 1, 1)
    embedding = encoder(batch)
    assert embedding.shape == (4, 5)
    assert torch.isfinite(embedding).all()

    unsigned_encoder = EdgeBranchEncoder(
        input_dim=3,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        use_signed_edge_separation=False,
    )
    unsigned_embedding = unsigned_encoder(batch)
    unsigned_embedding.sum().backward()
    assert unsigned_embedding.shape == (4, 5)
    assert torch.isfinite(unsigned_embedding).all()
    assert unsigned_encoder.edge_encoder[0].weight.grad is not None

    low_rank_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_projection_rank=3,
    )
    assert low_rank_encoder.edge_projection_rank == 3
    assert isinstance(low_rank_encoder.edge_encoder[0], torch.nn.Sequential)
    low_rank_embedding = low_rank_encoder(batch)
    low_rank_embedding.sum().backward()
    assert low_rank_embedding.shape == (4, 5)
    assert torch.isfinite(low_rank_embedding).all()
    assert low_rank_encoder.edge_encoder[0][0].weight.grad is not None

    dual_stream_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        use_dual_stream_signed_mlp=True,
    )
    dual_stream_embedding = dual_stream_encoder(batch)
    dual_stream_embedding.sum().backward()
    assert dual_stream_embedding.shape == (4, 5)
    assert torch.isfinite(dual_stream_embedding).all()
    assert dual_stream_encoder.positive_edge_encoder[0].weight.grad is not None
    assert dual_stream_encoder.negative_edge_encoder[0].weight.grad is not None
    assert dual_stream_encoder.signed_fusion_gate[0].weight.grad is not None

    profile_encoder = ROIProfileAttentionEncoder(
        num_nodes=3,
        embedding_dim=5,
        profile_dim=4,
        num_heads=2,
        dropout=0.1,
    )
    profile_embedding = profile_encoder(batch)
    profile_embedding.sum().backward()
    assert profile_embedding.shape == (4, 5)
    assert torch.isfinite(profile_embedding).all()
    assert profile_encoder.profile_encoder[0].weight.grad is not None

    try:
        EdgeBranchEncoder(
            input_dim=6,
            hidden_dim=8,
            embedding_dim=5,
            edge_projection_rank=6,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid edge_projection_rank to fail")

    baseline_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
    )
    residual_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        use_edge_residual=True,
        edge_residual_hidden_dim=4,
        edge_residual_scale=0.25,
    )
    residual_encoder.load_state_dict(baseline_encoder.state_dict(), strict=False)
    baseline_encoder.eval()
    residual_encoder.eval()
    with torch.no_grad():
        baseline_embedding = baseline_encoder(batch)
        residual_embedding = residual_encoder(batch)
    assert torch.allclose(baseline_embedding, residual_embedding, atol=1e-6)

    residual_encoder.train()
    residual_embedding = residual_encoder(batch)
    residual_embedding.sum().backward()
    assert residual_encoder.edge_residual[-1].weight.grad is not None
    assert torch.isfinite(residual_encoder.edge_residual[-1].weight.grad).all()
    assert residual_encoder.edge_residual[-1].weight.grad.abs().sum() > 0

    zero_dropout_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_dropout=0.0,
    )
    zero_dropout_encoder.load_state_dict(baseline_encoder.state_dict())
    baseline_encoder.eval()
    zero_dropout_encoder.eval()
    with torch.no_grad():
        baseline_embedding = baseline_encoder(batch)
        zero_dropout_embedding = zero_dropout_encoder(batch)
    assert torch.allclose(
        baseline_embedding,
        zero_dropout_embedding,
        atol=1e-6,
    )

    edge_dropout_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_dropout=0.2,
    )
    edge_dropout_encoder.train()
    dropout_embedding = edge_dropout_encoder(batch)
    dropout_embedding.sum().backward()
    assert dropout_embedding.shape == (4, 5)
    assert torch.isfinite(dropout_embedding).all()

    full_topk_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_topk_ratio=1.0,
    )
    full_topk_encoder.load_state_dict(baseline_encoder.state_dict())
    baseline_encoder.eval()
    full_topk_encoder.eval()
    with torch.no_grad():
        baseline_embedding = baseline_encoder(batch)
        full_topk_embedding = full_topk_encoder(batch)
    assert torch.allclose(
        baseline_embedding,
        full_topk_embedding,
        atol=1e-6,
    )

    sparse_topk_encoder = EdgeBranchEncoder(
        input_dim=6,
        hidden_dim=8,
        embedding_dim=5,
        dropout=0.1,
        edge_topk_ratio=0.5,
    )
    sparse_topk_encoder.train()
    sparse_topk_embedding = sparse_topk_encoder(batch)
    sparse_topk_embedding.sum().backward()
    assert sparse_topk_embedding.shape == (4, 5)
    assert torch.isfinite(sparse_topk_embedding).all()

    print("Edge encoder test passed.")
    print("Node summary:", tuple(summary.shape))
    print("Embedding:", tuple(embedding.shape))


if __name__ == "__main__":
    main()
