import torch
import torch.nn.functional as F
import numpy as np


def safe_atan2(y, x, eps=1e-8):
    """数值稳定的atan2"""
    # 避免0/0的情况
    x = torch.clamp(x, min=-1e6, max=1e6)
    y = torch.clamp(y, min=-1e6, max=1e6)
    return torch.atan2(y, x)


def angular_gradient_consistency_loss(
    conditions, text_emb, conditional_text_pos, model=None, alpha=0.15
):
    """
    只用已有的modulation结果 + 1次额外modulation
    目的是确保在condition space中，相同角度的condition，效果方向应该相似
    例如：
    - 如果条件1和条件2在condition space中角度相近，那么它们的效果方向应该相似
    """
    batch_size = len(conditions)

    # 计算所有condition的角度
    angles = safe_atan2(conditions[:, 1], conditions[:, 0])

    # 在当前batch中找角度相邻的pairs
    # 不需要全局排序，只需要找"最接近"的邻居
    angle_diff = torch.abs(angles.unsqueeze(0) - angles.unsqueeze(1))
    angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)  # 处理周期性

    # # 对每个sample，找到角度最接近的邻居（排除自己）
    # angle_diff.fill_diagonal_(float("inf"))

    # 修复：不用fill_diagonal_，用mask。不然会造成inplace操作，导致错误
    mask = torch.eye(len(conditions), device=angle_diff.device).bool()
    angle_diff = angle_diff.masked_fill(mask, float("inf"))

    nearest_neighbors = angle_diff.argmin(dim=1)

    # === 新增的1次modulation ===
    # 用每个样本的text，配对最接近邻居的condition
    neighbor_conditions = conditions[nearest_neighbors]
    conditional_text_neighbor = model.combine(text_emb, None, neighbor_conditions)

    # === 计算一致性 ===
    # 当前condition的效果方向
    delta_current = conditional_text_pos - text_emb

    # 邻居condition的效果方向
    delta_neighbor = conditional_text_neighbor - text_emb

    # 方向应该相似（角度接近的condition，效果方向也应该接近）
    consistency = F.cosine_similarity(delta_current, delta_neighbor, dim=-1)

    # 获取权重时也要避免inplace
    angle_to_neighbor = angle_diff.gather(1, nearest_neighbors.unsqueeze(1)).squeeze(1)
    angle_weights = torch.exp(-angle_to_neighbor)

    L_angular = ((1 - consistency) * angle_weights).mean()

    return alpha * L_angular


def rotation_semantic_orthogonality_loss(
    conditions, text_emb, conditional_text_pos, n_anchors=4, model=None, alpha=0.15
):
    """
    旋转不变性正则化
    用batch内的样本作为anchor points
    """
    batch_size = len(conditions)

    # 1. 在当前batch中选择n_anchors个"代表性"角度
    angles = torch.atan2(conditions[:, 1], conditions[:, 0])

    # 均匀分割角度空间，每个区间选一个代表
    angle_bins = torch.linspace(0, 2 * np.pi, n_anchors + 1)
    anchor_indices = []

    for i in range(n_anchors):
        bin_start, bin_end = angle_bins[i], angle_bins[i + 1]
        in_bin = (angles >= bin_start) & (angles < bin_end)

        if in_bin.any():
            # 选择这个bin中半径中等的一个（代表性）
            bin_conditions = conditions[in_bin]
            radii = torch.norm(bin_conditions, dim=1)
            median_idx = torch.argsort(radii)[len(radii) // 2]
            original_idx = in_bin.nonzero()[median_idx].item()
            anchor_indices.append(original_idx)

    if len(anchor_indices) < 2:
        return torch.tensor(0.0)  # batch太小，跳过

    # 2. === 新增的1次modulation ===
    # 用一个统一的"平均text"测试所有anchor conditions
    avg_text = text_emb.mean(dim=0, keepdim=True)

    anchor_conditions = conditions[anchor_indices]
    anchor_modulations = model.combine(
        avg_text.repeat(len(anchor_conditions), 1), None, anchor_conditions
    )

    # 3. 计算anchor之间的语义正交性
    # 相邻anchor的效果应该有明确差异
    anchor_deltas = anchor_modulations - avg_text

    L_ortho = 0
    for i in range(len(anchor_indices)):
        for j in range(i + 1, len(anchor_indices)):
            # 计算角度差
            angle_i = safe_atan2(anchor_conditions[i, 1], anchor_conditions[i, 0])
            angle_j = safe_atan2(anchor_conditions[j, 1], anchor_conditions[j, 0])
            angle_diff = torch.abs(angle_i - angle_j)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            # 语义相似度
            semantic_sim = F.cosine_similarity(
                anchor_deltas[i : i + 1], anchor_deltas[j : j + 1]
            )

            # 目标：语义相似度 ≈ cos(角度差)
            target_sim = torch.cos(angle_diff)
            L_ortho += (semantic_sim - target_sim).pow(2)

    return alpha * L_ortho / (len(anchor_indices) * (len(anchor_indices) - 1) / 2)


def manifold_smoothness_loss_sparse(
    conditions, text_emb, conditional_text_pos, k=3, model=None, alpha=0.1
):
    """
    只用batch内的k近邻，适合稀疏采样
    """
    batch_size = len(conditions)

    # 1. 在condition space中计算距离矩阵
    dist_matrix = torch.cdist(conditions, conditions)

    mask = torch.eye(batch_size, device=dist_matrix.device).bool()
    dist_matrix = dist_matrix.masked_fill(mask, float("inf"))

    # 2. 每个样本找k个最近邻
    _, neighbor_indices = torch.topk(dist_matrix, k, largest=False, dim=1)

    # 3. === 新增的1次modulation ===
    # 对每个样本，用它的text配对它邻居的conditions
    # 我们随机选一个邻居（减少计算）
    random_neighbor_idx = torch.randint(0, k, (batch_size,))
    selected_neighbors = neighbor_indices[torch.arange(batch_size), random_neighbor_idx]

    neighbor_conditions = conditions[selected_neighbors]
    conditional_text_from_neighbor = model.combine(text_emb, None, neighbor_conditions)

    # 4. 计算平滑性
    # 当前condition的效果
    delta_current = conditional_text_pos - text_emb

    # 邻居condition的效果（在同一个text上）
    delta_neighbor = conditional_text_from_neighbor - text_emb

    # 应该相似（近邻在condition space中，效果也应该接近）
    smoothness = F.cosine_similarity(delta_current, delta_neighbor, dim=-1)

    # 5. 加权：距离越近，约束越强
    distances = dist_matrix.gather(1, selected_neighbors.unsqueeze(1)).squeeze(1)
    distances = torch.clamp(distances, min=1e-8, max=10.0)
    weights = torch.exp(-distances + 1e-8)  # 距离近的权重大

    L_smooth_weighted = ((1 - smoothness) * weights).mean()

    return alpha * L_smooth_weighted


def radius_monotonicity_loss(condition_2d, text_emb, model, n_samples=50, alpha=0.1):
    """
    确保：相同角度，半径越大，调制效果越强

    核心思想：
    - 固定角度θ
    - 采样不同半径 r1 < r2
    - 要求：||text_r2 - text_orig|| > ||text_r1 - text_orig||
    """
    batch_size = len(condition_2d)

    if batch_size < 2:
        return torch.tensor(0.0, device=condition_2d.device)

    # 随机采样pairs
    n_samples = min(n_samples, batch_size * (batch_size - 1) // 2)

    loss = 0
    count = 0

    for _ in range(n_samples):
        # 随机选两个conditions
        idx = torch.randperm(batch_size)[:2]
        c1, c2 = condition_2d[idx[0]], condition_2d[idx[1]]

        # 计算角度和半径
        angle1 = torch.atan2(c1[1], c1[0])
        angle2 = torch.atan2(c2[1], c2[0])
        radius1 = torch.norm(c1)
        radius2 = torch.norm(c2)

        # 只考虑角度相近的pairs（容忍度：15度）
        angle_diff = torch.abs(angle1 - angle2)
        angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)

        if angle_diff > np.pi / 12:  # 15度
            continue

        # 确定哪个半径更大
        if radius1 > radius2:
            larger_idx, smaller_idx = idx[0], idx[1]
            r_large, r_small = radius1, radius2
        else:
            larger_idx, smaller_idx = idx[1], idx[0]
            r_large, r_small = radius2, radius1

        # 如果半径差异太小，跳过
        if r_large - r_small < 0.1:
            continue

        # 计算调制效果
        text_orig = text_emb[larger_idx : larger_idx + 1]  # 用一个作为参考

        cond_large = condition_2d[larger_idx : larger_idx + 1]
        cond_small = condition_2d[smaller_idx : smaller_idx + 1]
        text_large = model.combine(text_orig, None, cond_large)
        text_small = model.combine(text_orig, None, cond_small)

        # 计算变化幅度
        delta_large = torch.norm(text_large - text_orig, dim=1)
        delta_small = torch.norm(text_small - text_orig, dim=1)

        # 损失：半径大的应该变化更大
        # 使用margin来避免过于严格
        margin = 0.05
        violation = F.relu(delta_small - delta_large + margin)

        loss += violation
        count += 1

    return alpha * loss / max(count, 1)
