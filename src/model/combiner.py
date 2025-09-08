# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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

    def freeze_switch_modulation(self):
        pass

    def freeze_all_modulation(self):
        for p in self.gamma.parameters():
            p.requires_grad = False
        for p in self.beta.parameters():
            p.requires_grad = False
        self.frozen = True

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
