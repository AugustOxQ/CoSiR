import torch

from src.model import CombinerGated
from src.model import CoSiRModel


def test_combiner():
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


def test_cosirmodel():
    model = CoSiRModel()
    image_input = torch.randn(2, 3, 224, 224)
    text_input = torch.randn(2, 77, 512)
    label = torch.randn(2, 32)
    output = model(image_input, text_input, label)
    print(output.shape)


def main():
    test_combiner()
    test_cosirmodel()


if __name__ == "__main__":
    main()
