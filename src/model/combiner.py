# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque
from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn


import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be greater than or equal to 1.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 1. define internal sequential layers
        layers = []

        # the first layer in the block
        if num_layers == 1:
            # if there is only 1 layer, directly from input_dim to output_dim
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # otherwise, from input_dim to hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())  # activation function
            layers.append(
                nn.LayerNorm(hidden_dim)
            )  # normalization (optional, but common)

            # intermediate layers (if any)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))

            # the last layer in the block
            layers.append(nn.Linear(hidden_dim, output_dim))

        # 2. wrap internal network
        self.dense_layers = nn.Sequential(*layers)

        # 3. define projection for skip connection
        # if the input and output dimensions are different, a linear layer is needed to match the dimensions, so that addition can be performed.
        if input_dim != output_dim:
            self.skip_connection_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection_proj = (
                nn.Identity()
            )  # if the dimensions are the same, use the identity mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # calculate the internal network output (F(x))
        residual = self.dense_layers(x)

        # calculate the skip connection projection (W_s * x)
        # this step is to ensure dimension matching
        shortcut = self.skip_connection_proj(x)

        # ResNet core operation: F(x) + W_s * x
        out = residual + shortcut

        # usually the last layer in the ResNet block has an activation function (but in actual networks, this is usually at the input of the next block or at the end of the network)
        # here we don't add the final activation, so it can be flexibly used as part of a larger network.
        return out


class FixedSizeQueue:
    def __init__(self, max_size: int) -> None:
        """Initialize a FixedSizeQueue with a given maximum size.

        Args:
            max_size: The maximum size of the queue.

        The queue is implemented as a deque with a maximum length equal to max_size.
        When the queue is full and another element is added, the oldest element is removed.
        """
        self.queue = deque(maxlen=max_size)

    def add(self, number: float) -> None:
        """Add a number to the end of the queue. If the queue is full, remove the oldest element
        first.

        Args:
            number: The number to add to the queue.
        """
        self.queue.append(number)

    def get(self) -> list[str]:
        """Return a list of the current elements in the queue as strings, each formatted to 2
        decimal places.

        Returns:
            A list of strings, where each string is a number from the queue formatted to 2 decimal places.
        """
        return [f"{num:.2f}" for num in self.queue]

    def get_newest(self) -> float:
        """Return the newest element in the queue.

        If the queue is not empty, return the last element in the deque.
        If the queue is empty, return -1.0.

        Returns:
            The newest element in the queue, or -1.0 if the queue is empty.
        """
        if self.queue:
            return self.queue[-1]  # The last element in the deque
        else:
            return -1.0  # In case the queue is empty


class Combiner_basic(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    @torch.jit.export
    def forward(
        self, text_features: Tensor, text_full: Tensor, label_features: Tensor
    ) -> Tensor:
        """Combine the text features and label features using attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        text_projected_features = self.dropout1(
            F.relu(self.text_projection_layer(text_features))
        )
        label_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(label_features))
        )

        raw_combined_features = torch.cat(
            (text_projected_features, label_projected_features), -1
        )
        combined_features = self.dropout3(
            F.relu(self.combiner_layer(raw_combined_features))
        )

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        # print(dynamic_scalar.shape) # (batch, 1)
        self.scalar.add(dynamic_scalar.mean().item())
        # print(self.scalar.get())

        # # Option1: Output is a combination of combined_featured and text_features and label_projected_features
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_projected_features
        )

        # Option2: Output is a combination of combined_featured and text_features
        # output = (
        #     dynamic_scalar * self.output_layer(combined_features)
        #     + (1 - dynamic_scalar) * text_features
        # )

        # Option3: Output is combined_features
        # output = self.output_layer(combined_features) + text_features

        return F.normalize(output)


class Combiner_new(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        label_dim: int = 2,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.5,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.label_decoder = GeLUNetGradual(
            input_dim=label_dim,
            output_dim=clip_feature_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # self.text_weighter = GeLUNetGradual(
        #     input_dim=label_dim,
        #     output_dim=clip_feature_dim,
        #     num_layers=num_layers,
        #     dropout=dropout,
        # )

        # for param in self.label_decoder.network[-1].parameters():
        #     param.data.zero_()

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    def forward(
        self,
        text_features: Tensor,
        text_full: Optional[Tensor],
        label_features: Tensor,
        return_delta: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        delta = self.label_decoder(label_features)

        combined = text_features + delta

        if return_delta:
            return F.normalize(combined), delta
        return F.normalize(combined)


class CombinerGated(nn.Module):
    """Combiner module using gated residual + additive label shift."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
        warm_up_epoch: int = 5,
        scale_init: float = 1,
    ) -> None:
        super().__init__()

        # Fixed orthogonal projection
        self.label_proj_layer = nn.Linear(label_dim, projection_dim)
        nn.init.orthogonal_(self.label_proj_layer.weight)
        for param in self.label_proj_layer.parameters():
            param.requires_grad = False

        self.warm_up_epoch = warm_up_epoch
        self.scalar = FixedSizeQueue(2)

        # Learnable channel-wise scale (optional)
        self.scale = nn.Parameter(torch.ones(projection_dim) * scale_init)

        gamma_middle = nn.Linear(
            projection_dim + clip_feature_dim + label_dim, clip_feature_dim
        )
        nn.init.zeros_(gamma_middle.weight)
        nn.init.zeros_(gamma_middle.bias)

        # Gating and additive shift (trainable)
        self.gamma = nn.Sequential(
            nn.LayerNorm(projection_dim + clip_feature_dim + label_dim),
            gamma_middle,
            nn.Tanh(),  # Output around [-1, 1]
        )

        beta_middle = nn.Linear(projection_dim, clip_feature_dim)
        beta_middle.weight.data.copy_(torch.eye(clip_feature_dim))
        nn.init.zeros_(beta_middle.bias)

        # Beta: additive semantic shift
        self.beta = nn.Sequential(
            nn.LayerNorm(projection_dim),
            beta_middle,
        )

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    def forward(
        self,
        text_features: Tensor,  # [B, 512]
        text_full: Tensor,  # unused
        label_features: Tensor,  # [B, label_dim]
        epoch: int,
        return_label_proj: bool = False,
    ):

        label_proj = self.label_proj_layer(label_features)
        raw_cat = torch.cat(
            (label_proj, text_features, label_features), dim=-1
        )  # [B, 512 + 512]

        # Gated modulation
        gamma = self.gamma(raw_cat)  # [B, 512]
        beta = self.beta(label_proj)  # [B, 512]
        combined = text_features + gamma * text_features + beta

        self.scalar.add(gamma.mean().item())
        self.scalar.add(beta.mean().item())

        if return_label_proj:
            return F.normalize(combined, dim=-1), label_proj
        else:
            return F.normalize(combined, dim=-1)


class GeLUNet(nn.Module):
    """
    A lightweight GeLU network with configurable depth.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Total number of linear layers (including input and output)
        dropout: Dropout probability (default: 0.1)
        use_output_activation: Whether to apply activation on output layer (default: False)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_output_activation: bool = False,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        layers: list[nn.Module] = []

        # First layer
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # Special case: only 1 layer
            layers = [nn.Linear(input_dim, output_dim)]

        # Optional output activation
        if use_output_activation:
            layers.extend(
                [
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                ]
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GeLUNetGradual(nn.Module):
    """
    A GeLU network whose hidden widths are interpolated geometrically (log-space)
    between input_dim and output_dim, so the network gradually expands or contracts
    instead of using a fixed hidden_dim throughout.

    Example: input_dim=2, output_dim=512, num_layers=4
        layer dims: 2 -> 8 -> 64 -> 512  (roughly evenly spaced in log-space)

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        num_layers: Total number of linear layers (including input and output)
        dropout: Dropout probability (default: 0.1)
        use_output_activation: Whether to apply LayerNorm+GELU on the output layer (default: False)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_output_activation: bool = False,
    ):
        import math

        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        # Compute output dim for each layer via geometric interpolation.
        # t goes from 1/N to 1, so dims[-1] == output_dim exactly.
        dims: list[int] = []
        for i in range(num_layers):
            t = (i + 1) / num_layers
            d = int(
                round(
                    math.exp(math.log(input_dim) * (1 - t) + math.log(output_dim) * t)
                )
            )
            dims.append(max(d, 1))
        dims[-1] = output_dim  # guarantee exact match

        layers: list[nn.Module] = []
        in_d = input_dim
        for i, out_d in enumerate(dims):
            layers.append(nn.Linear(in_d, out_d))
            is_last = i == len(dims) - 1
            if not is_last:
                layers.extend([nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout)])
            elif use_output_activation:
                layers.extend([nn.LayerNorm(out_d), nn.GELU()])
            in_d = out_d

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
