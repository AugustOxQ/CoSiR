# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque

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


class CombinerPolar(nn.Module):
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

        self.radius_to_scale = ResNetBlock(
            input_dim=1, hidden_dim=64, output_dim=1, num_layers=4
        )
        self.angle_to_rotate = ResNetBlock(
            input_dim=1, hidden_dim=128, output_dim=clip_feature_dim, num_layers=4
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

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

        batch_size = text_features.shape[0]

        label_proj = self.label_proj_layer(label_features)

        # Change to polar coordinates
        radius = torch.norm(label_features, dim=1, keepdim=True)
        angle = torch.atan2(label_features[:, 1:], label_features[:, :1])

        # Redicus to scale
        scale_raw = self.radius_to_scale(radius)
        scale_raw = self.relu(scale_raw)
        scale = 0.8 + 0.4 * scale_raw  # This project to [0.8, 1.2]

        # Angle to rotate
        rotation_axis = self.angle_to_rotate(angle)
        rotation_axis = self.tanh(rotation_axis)
        rotation_axis = F.normalize(rotation_axis, dim=-1)

        # Projection
        text_norm = F.normalize(text_features, dim=-1)
        dot_product = (text_norm * rotation_axis).sum(dim=-1, keepdim=True)

        # Gram-Schmidt orthogonalization
        axis_component = dot_product * rotation_axis
        perpendicular = text_norm - axis_component
        perpendicular = F.normalize(perpendicular, dim=-1)

        # Rotate
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        # Angle after rotation
        rotated_direction = cos_angle * text_norm + sin_angle * perpendicular

        # Combine: scale + rotate
        text_magnitute = torch.norm(text_norm, dim=-1, keepdim=True).clamp(min=1e-8)
        combined = scale * rotated_direction + text_magnitute

        if return_label_proj:
            return F.normalize(combined, dim=-1), label_proj
        else:
            return F.normalize(combined, dim=-1)


class CombinerSimplePolar_enhance(nn.Module):
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
        self.complex_dim = clip_feature_dim // 2
        self.clip_feature_dim = clip_feature_dim

        self.radius_to_scale = ResNetBlock(
            input_dim=1, hidden_dim=128, output_dim=256, num_layers=2
        )
        self.angle_to_rotate = ResNetBlock(
            input_dim=1, hidden_dim=64, output_dim=256, num_layers=4
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

        batch_size = text_features.shape[0]
        label_proj = self.label_proj_layer(label_features)

        # Change to polar coordinates
        radius = torch.norm(label_features, dim=1, keepdim=True)  # [B, 1]
        angle = torch.atan2(
            label_features[:, 1:2], label_features[:, 0:1]
        )  # [B, 1] #TODO: print out and check the difference, then check the gradient

        # Interpret text_emb as complex numbers
        # [B, 512] -> [B, 256, 2] (real, imag)
        text_complex = text_features.view(batch_size, self.complex_dim, 2)
        text_real = text_complex[:, :, 0]  # [B, 256]
        text_imag = text_complex[:, :, 1]  # [B, 256]

        # Generate modulated complex numbers (polar form)
        phase_shifts = self.angle_to_rotate(angle)  # [B, 256]
        magnitudes = self.radius_to_scale(radius)  # [B, 256]
        magnitudes = 0.5 + magnitudes  # [0.5, 1.5]

        # Complex multiplication: (a+bi) * (r·e^(iθ)) = r·(a·cos(θ) - b·sin(θ) + i(a·sin(θ) + b·cos(θ)))
        cos_phase = torch.cos(phase_shifts)
        sin_phase = torch.sin(phase_shifts)

        # Apply complex multiplication
        modulated_real = magnitudes * (text_real * cos_phase - text_imag * sin_phase)
        modulated_imag = magnitudes * (text_real * sin_phase + text_imag * cos_phase)

        # Convert back to real representation
        modulated_complex = torch.stack(
            [modulated_real, modulated_imag], dim=2
        )  # [B, 256, 2]
        combined = modulated_complex.view(batch_size, self.clip_feature_dim)  # [B, 512]

        if return_label_proj:
            return F.normalize(combined, dim=-1), label_proj
        else:
            return F.normalize(combined, dim=-1)


class CombinerSimplePolar(nn.Module):
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
        self.clip_feature_dim = clip_feature_dim

        self.radius_to_scale = ResNetBlock(
            input_dim=1, hidden_dim=64, output_dim=1, num_layers=2
        )
        self.angle_to_rotate = ResNetBlock(
            input_dim=1, hidden_dim=64, output_dim=1, num_layers=2
        )

        self.max_rotation_angle = torch.pi / 6  # 30度

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

        batch_size = text_features.shape[0]
        label_proj = self.label_proj_layer(label_features)

        # condition: [B, 2]
        # text_features: [B, 512]

        # 1. 转极坐标
        radius = torch.norm(label_features, dim=1, keepdim=True)  # [B, 1]
        angle = torch.atan2(label_features[:, 1:2], label_features[:, 0:1])  # [B, 1]

        # 2. Radius -> Scale (范围 [0.8, 1.2])
        scale_factor = 0.8 + 0.4 * self.radius_to_scale(radius)  # [B, 1]

        # 3. Angle -> Rotation strength
        rotation_strength = self.angle_to_rotate(angle)  # [B, 1], 范围[-1,1]

        # 4. Project condition to feature space (frozen)
        condition_proj = self.label_proj_layer(label_features)  # [B, 512]
        condition_proj = F.normalize(condition_proj, dim=-1)

        # 5. 简单的旋转: 在text和condition_proj张成的平面内旋转
        text_norm = F.normalize(text_features, dim=-1)

        # 计算旋转角度 (由rotation_strength控制, 最大±30度)
        rotation_angle = rotation_strength * self.max_rotation_angle  # [B, 1]

        # 在text_norm和condition_proj的平面内旋转
        # 使用简单的线性插值近似
        cos_theta = torch.cos(rotation_angle)
        sin_theta = torch.sin(rotation_angle)

        rotated = cos_theta * text_norm + sin_theta * condition_proj
        rotated = F.normalize(rotated, dim=-1)

        # 6. Apply scale (保持原始magnitude)
        text_magnitude = torch.norm(text_features, dim=-1, keepdim=True)
        combined = scale_factor * text_magnitude * rotated

        if return_label_proj:
            return F.normalize(combined, dim=-1), label_proj
        else:
            return F.normalize(combined, dim=-1)


class CombinerSimplePolar_noparam(nn.Module):
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

        self.scalar = FixedSizeQueue(2)
        self.clip_feature_dim = clip_feature_dim

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

        batch_size = text_features.shape[0]
        label_proj = self.label_proj_layer(label_features)

        # condition: [B, 2]
        # text_features: [B, 512]

        # 1. 转极坐标
        radius = torch.norm(label_features, dim=1, keepdim=True)  # [B, 1]
        angle = torch.atan2(label_features[:, 1:2], label_features[:, 0:1])  # [B, 1]

        # 2. Scale: 直接用radius (无参数)
        scale_factor = 0.8 + 0.4 * torch.sigmoid(radius)  # [B, 1]

        # 3. Rotation: 用condition在前两个维度做小扰动
        text_norm = F.normalize(text_features, dim=-1)

        # 在特征空间的前两个维度上应用condition
        perturbation = torch.zeros_like(text_features)
        # 旋转强度由angle的tanh控制
        rotation_strength = torch.tanh(angle) * 0.1  # 小扰动
        perturbation[:, 0] = label_features[:, 0] * rotation_strength.squeeze()
        perturbation[:, 1] = label_features[:, 1] * rotation_strength.squeeze()

        rotated = text_norm + perturbation
        rotated = F.normalize(rotated, dim=-1)

        # 4. Apply scale
        text_magnitude = torch.norm(text_features, dim=-1, keepdim=True)
        combined = scale_factor * text_magnitude * rotated

        if return_label_proj:
            return F.normalize(combined, dim=-1), label_proj
        else:
            return F.normalize(combined, dim=-1)
