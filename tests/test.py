import torch

from src.model.combiner import CombinerGated


def test_forward_variables_shape_and_type():
    combiner = CombinerGated(
        clip_feature_dim=512,
        projection_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        label_dim=32,
    )
    text_features = torch.randn(2, 512)
    text_full = torch.randn(2, 77, 512)

    for i in range(20):
        label_features = torch.randn(2, 32)
        output = combiner.forward(text_features, text_full, label_features, 0)
        print(output.shape, f"{output[0][:10].tolist()}")  # type: ignore


if __name__ == "__main__":
    test_forward_variables_shape_and_type()
