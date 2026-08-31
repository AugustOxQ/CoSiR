"""
Utility to load and inspect the 5k×5k annotation matrix.

Example:
    from load_annotations import AnnotationMatrix
    ann = AnnotationMatrix("/data/SSD2/annotations/redcaps_5k5k/annotation_matrix.npz")
    ann.summary()
    label = ann.get(i, j)             # 1=good, 0=bad, -1=unannotated
    good_caps = ann.good_captions_for_image(i)
"""

import numpy as np
from pathlib import Path


class AnnotationMatrix:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.matrix: np.ndarray = data["matrix"]          # (N, N) int8
        self.image_ids: np.ndarray = data["image_ids"]    # (N,) str
        self.captions: np.ndarray = data["captions"]      # (N,) str
        self.image_rel_paths: np.ndarray = data["image_rel_paths"]  # (N,) str
        self.n = self.matrix.shape[0]

    def get(self, img_idx: int, cap_idx: int) -> int:
        """Return annotation for (image i, caption j): 1=good, 0=bad, -1=unannotated."""
        return int(self.matrix[img_idx, cap_idx])

    def good_captions_for_image(self, img_idx: int) -> list[str]:
        """Return all captions annotated as good (1) for image i."""
        mask = self.matrix[img_idx] == 1
        return list(self.captions[mask])

    def good_images_for_caption(self, cap_idx: int) -> list[str]:
        """Return all image_ids annotated as good (1) for caption j."""
        mask = self.matrix[:, cap_idx] == 1
        return list(self.image_ids[mask])

    def completion_fraction(self) -> float:
        total = self.n * self.n
        annotated = int((self.matrix != -1).sum())
        return annotated / total

    def summary(self):
        total = self.n * self.n
        good = int((self.matrix == 1).sum())
        bad = int((self.matrix == 0).sum())
        unann = int((self.matrix == -1).sum())
        diag_good = int((self.matrix.diagonal() == 1).sum())
        print(
            f"AnnotationMatrix ({self.n}×{self.n})\n"
            f"  Total pairs:        {total:,}\n"
            f"  Good (1):           {good:,}  ({100*good/total:.1f}%)\n"
            f"  Bad (0):            {bad:,}  ({100*bad/total:.1f}%)\n"
            f"  Unannotated (-1):   {unann:,}  ({100*unann/total:.1f}%)\n"
            f"  GT diagonal good:   {diag_good}/{self.n}  "
            f"({100*diag_good/self.n:.1f}% of original pairs annotated good)\n"
        )
