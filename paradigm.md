# Spatial Collapse & Residual Projection Framework: Paradigms 1 & 2

This document extracts the formal mathematical foundations, algorithmic derivations, and PyTorch implementations for **Paradigm 1** and **Paradigm 2** from `Low-Rank Residual Decomposition for Modeling.pdf`.

---

## 1. The Common Foundation: Spatial Domain Variance Pruning

Before applying any matrix decomposition or linear projection, patient scans undergo spatial variance pruning to collapse invariant anatomical and scanner background pixels.

### 1.1 Mathematical Formulation
Let $X \in \mathbb{R}^{N \times P}$ represent a cohort of $N$ registered medical image slices flattened across $P = H \times W$ spatial pixel coordinates.

Because images are spatially registered to a common anatomical coordinate system (e.g., via ANTs / SyN to an SRI24 / MNI template), a large percentage of coordinates $j \in \{1, \dots, P\}$ correspond to invariant non-informative scanner background or static skull/boundary tissue.

For every spatial coordinate $j$, compute its cross-patient variance across all $N$ patient samples:
$$\sigma_j^2 = \frac{1}{N - 1} \sum_{i=1}^N (X_{ij} - \mu_j)^2$$

Define the spatial domain pruning operator $\mathcal{P}_\tau: \mathbb{R}^P \to \mathbb{R}^{P'}$ with variance cutoff $\tau > 0$:
$$X_{\text{pruned}} = X \mathcal{P}_\tau^T \in \mathbb{R}^{N \times P'}$$

where:
$$P' = \sum_{j=1}^P \mathbb{I}(\sigma_j^2 \ge \tau) \ll P$$

**Benefits**:
* Drops 50% to 80% of spatial dimensions before decomposition.
* Reduces SVD/inversion complexity from $\mathcal{O}(N \cdot P^2)$ down to $\mathcal{O}(N \cdot P'^2)$.
* Prevents zero-variance scanner background noise from distorting low-rank subspaces.

### 1.2 PyTorch Implementation
```python
import torch
from torch import Tensor
from typing import Optional, Union, Tuple

class CommonSpatialPruner:
    """
    Step 1 Engine: Spatial Feature Variance Pruning.
    Collapses invariant feature columns (e.g., static background) across registered cohort.
    """
    def __init__(self, var_threshold: float = 1e-4):
        self.var_threshold = var_threshold
        self.active_mask: Optional[Tensor] = None
        self.H: Optional[int] = None
        self.W: Optional[int] = None
        self.P_total: Optional[int] = None
        self.P_active: Optional[int] = None

    def fit_transform(self, X_stack: Tensor) -> Tensor:
        """
        Args:
            X_stack: Tensor of shape (N, H, W) or (N, C, H, W) or (N, P)
        Returns:
            X_pruned: Tensor of shape (N, P_active)
        """
        if X_stack.ndim == 4:
            # Grayscale conversion: (N, C, H, W) -> (N, H, W)
            X_stack = 0.299 * X_stack[:, 0] + 0.587 * X_stack[:, 1] + 0.114 * X_stack[:, 2]

        if X_stack.ndim == 3:
            N, self.H, self.W = X_stack.shape
            self.P_total = self.H * self.W
            X_flat = X_stack.view(N, -1)
        else:
            N, self.P_total = X_stack.shape
            X_flat = X_stack

        # Compute column-wise variance across N patient scans
        col_vars = torch.var(X_flat, dim=0, unbiased=True)
        self.active_mask = col_vars > self.var_threshold
        self.P_active = int(self.active_mask.sum().item())

        X_pruned = X_flat[:, self.active_mask]
        return X_pruned

    def transform(self, X_query: Tensor) -> Tensor:
        """Projects incoming test scans into active coordinates: (B, C, H, W) -> (B, P_active)."""
        if X_query.ndim == 4:
            X_query = 0.299 * X_query[:, 0] + 0.587 * X_query[:, 1] + 0.114 * X_query[:, 2]
        B = X_query.shape[0] if X_query.ndim == 3 else 1
        X_flat = X_query.view(B, -1)
        return X_flat[:, self.active_mask.to(X_query.device)]

    def map_to_2d(self, X_pruned_vector: Tensor) -> Tensor:
        """Maps 1D pruned vectors (or 2D batch B x P') back to original spatial grid (B, H, W)."""
        is_single = X_pruned_vector.ndim == 1
        X_p = X_pruned_vector.unsqueeze(0) if is_single else X_pruned_vector
        B = X_p.shape[0]

        grid = torch.zeros(B, self.P_total, device=X_p.device, dtype=X_p.dtype)
        grid[:, self.active_mask.to(X_p.device)] = X_p
        spatial_grid = grid.view(B, self.H, self.W)
        return spatial_grid.squeeze(0) if is_single else spatial_grid
```

---

## 2. Paradigm 1: Fully Unsupervised Robust PCA (IALM-PCP)

### 2.1 Mathematical Formulation
Paradigm 1 operates under **zero supervision** on an unlabelled cohort of mixed patient scans (containing an unknown mix of normal and diseased images).

It models the active matrix $X_{\text{pruned}} \in \mathbb{R}^{N \times P'}$ as a superposition of a low-rank background $L$ (shared normal anatomy across patients) and a sparse matrix $S$ (pathological lesions):

$$\min_{L, S} \quad \|L\|_* + \lambda \|S\|_1 \quad \text{subject to} \quad X_{\text{pruned}} = L + S$$

Where:
* $\|L\|_* = \sum_i \sigma_i(L)$ is the nuclear norm (sum of singular values), enforcing low rank on shared anatomy.
* $\|S\|_1 = \sum_{i,j} |S_{ij}|$ is the $L_1$ norm, enforcing spatial sparsity on disease anomalies.
* $\lambda = \frac{1}{\sqrt{\max(N, P')}}$ is the standard theoretical regularization parameter.

### 2.2 Optimization via Inexact Augmented Lagrange Multipliers (IALM)
The augmented Lagrangian function is:
$$\mathcal{L}(L, S, Y, \mu) = \|L\|_* + \lambda \|S\|_1 + \langle Y, X_{\text{pruned}} - L - S \rangle + \frac{\mu}{2} \|X_{\text{pruned}} - L - S\|_F^2$$

Where $Y$ is the Lagrange multiplier matrix and $\mu > 0$ is the penalty parameter.

At each iteration $k$, the IALM solver alternates two closed-form proximal operations:

1. **Sparse Matrix Update via Soft-Thresholding ($\mathcal{S}_\tau$)**:
   $$S^{(k+1)} = \mathcal{S}_{\frac{\lambda}{\mu_k}}\left(X_{\text{pruned}} - L^{(k)} + \frac{1}{\mu_k} Y^{(k)}\right)$$
   where $\mathcal{S}_\tau(x) = \text{sign}(x) \cdot \max(|x| - \tau, 0)$.

2. **Low-Rank Matrix Update via Singular Value Thresholding ($\mathcal{D}_\tau$)**:
   $$L^{(k+1)} = \mathcal{D}_{\frac{1}{\mu_k}}\left(X_{\text{pruned}} - S^{(k+1)} + \frac{1}{\mu_k} Y^{(k)}\right)$$
   where $\mathcal{D}_\tau(M) = U \cdot \mathcal{S}_\tau(\Sigma) \cdot V^T$.

3. **Dual Update & Penalty Scaling**:
   $$Z = X_{\text{pruned}} - L^{(k+1)} - S^{(k+1)}$$
   $$Y^{(k+1)} = Y^{(k)} + \mu_k Z$$
   $$\mu_{k+1} = \min(\rho \mu_k, \mu_{\max})$$
   where $\rho \approx 1.5$. Convergence occurs when $\|Z\|_F / \|X_{\text{pruned}}\|_F < \text{tol}$.

### 2.3 PyTorch Implementation
```python
import torch
from torch import Tensor
from typing import Tuple, Optional

class Paradigm1_RPCA:
    """
    Paradigm 1: Unsupervised Low-Rank (L) + Sparse Residual (S) Decomposition via IALM-PCP.
    No labels or clean normal references required.
    """
    def __init__(self, lmbda: Optional[float] = None, max_iter: int = 100, tol: float = 1e-6):
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _soft_threshold(X: Tensor, tau: float) -> Tensor:
        """Soft-thresholding operator S_tau(X) = sign(X) * max(|X| - tau, 0)"""
        return torch.sign(X) * torch.clamp(torch.abs(X) - tau, min=0.0)

    @staticmethod
    def _singular_value_threshold(X: Tensor, tau: float) -> Tensor:
        """Singular Value Thresholding D_tau(X) = U * S_tau(Sigma) * V^T"""
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S_thresh = torch.sign(S) * torch.clamp(torch.abs(S) - tau, min=0.0)
        return (U * S_thresh.unsqueeze(-2)) @ Vh

    def decompose(self, X_pruned: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decomposes X_pruned into:
            L (Low-rank background anatomy)
            S (Sparse pathological lesions)
        """
        N, P = X_pruned.shape
        lmbda = self.lmbda if self.lmbda is not None else 1.0 / (max(N, P) ** 0.5)

        norm_two = torch.linalg.matrix_norm(X_pruned, ord=2)
        norm_inf = torch.linalg.vector_norm(X_pruned, ord=float('inf')) / lmbda
        dual_norm = max(norm_two.item(), norm_inf.item())

        Y = X_pruned / dual_norm
        L = torch.zeros_like(X_pruned)
        S = torch.zeros_like(X_pruned)

        mu = 1.25 / norm_two.item()
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = torch.linalg.matrix_norm(X_pruned, ord='fro')

        for k in range(self.max_iter):
            # 1. Update Sparse Matrix S
            temp_S = X_pruned - L + (1.0 / mu) * Y
            S = self._soft_threshold(temp_S, lmbda / mu)

            # 2. Update Low-Rank Matrix L
            temp_L = X_pruned - S + (1.0 / mu) * Y
            L = self._singular_value_threshold(temp_L, 1.0 / mu)

            # 3. Update Dual Variables
            Z = X_pruned - L - S
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = torch.linalg.matrix_norm(Z, ord='fro') / d_norm
            if err < self.tol:
                break

        return L, S
```

---

## 3. Paradigm 2: Semi-Supervised Learned Healthy Linear Operator ($W^*$)

### 3.1 Mathematical Formulation
When a reference cohort of confirmed healthy patient scans $X_H \in \mathbb{R}^{N_H \times P'}$ is available, Paradigm 2 explicitly learns a spatial transformation matrix $W^* \in \mathbb{R}^{P' \times P'}$ that maps feature columns to feature columns across healthy anatomy:

$$W^* = \arg\min_W \frac{1}{2} \|X_H - X_H W\|_F^2 + \frac{\alpha}{2} \|W\|_F^2$$

This is solved in **closed-form** via Tikhonov-regularized Ridge Regression:
$$W^* = (X_H^T X_H + \alpha I)^{-1} X_H^T X_H$$

Where:
* $X_H^T X_H \in \mathbb{R}^{P' \times P'}$ is the spatial covariance/Gram matrix.
* $\alpha > 0$ ensures strict positive-definiteness and numerical stability.

### 3.2 Residual Projection for Test Scans
At inference time, for any query patient scan $x_{\text{query}} \in \mathbb{R}^{1 \times P'}$ (whether healthy or diseased):
1. **Reconstruct Predicted Healthy State**:
   $$\hat{x}_{\text{healthy}} = x_{\text{query}} W^*$$
2. **Extract Pathology Residual**:
   $$r = x_{\text{query}} - \hat{x}_{\text{healthy}} = x_{\text{query}} (I - W^*)$$

Where $(I - W^*)$ acts as the **healthy annihilation operator**:
* Any image adhering to the healthy anatomical manifold is mapped to $\approx 0$.
* Any pathological lesion that cannot be linearly explained by healthy tissue produces high-magnitude residual activations in $r$.

### 3.3 PyTorch Implementation
```python
import torch
from torch import Tensor
from typing import Tuple, Optional

class Paradigm2_LinearOperator:
    """
    Paradigm 2: Semi-Supervised Learned Healthy Operator W* and Residual Projection.
    Requires confirmed healthy reference scans (X_H) for fitting.
    """
    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.W: Optional[Tensor] = None

    def fit(self, X_healthy_pruned: Tensor) -> "Paradigm2_LinearOperator":
        """
        Fits W* = (X_H^T X_H + alpha * I)^(-1) X_H^T X_H
        Input shape: (N_H, P_pruned)
        """
        N_H, P = X_healthy_pruned.shape
        Gram = X_healthy_pruned.T @ X_healthy_pruned
        Reg = self.alpha * torch.eye(P, device=X_healthy_pruned.device, dtype=X_healthy_pruned.dtype)

        # Closed-form regularized solve: (Gram + Reg) W = Gram
        self.W = torch.linalg.solve(Gram + Reg, Gram)
        return self

    def transform(self, X_query_pruned: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Input shape: (B, P_pruned)
        Returns:
            X_hat: Reconstructed healthy projection (B, P_pruned)
            Residuals: Pathology anomaly signal r = x(I - W*) (B, P_pruned)
        """
        if self.W is None:
            raise RuntimeError("Operator not fitted! Call fit() with healthy scans first.")

        X_hat = X_query_pruned @ self.W
        residuals = X_query_pruned - X_hat
        return X_hat, residuals
```

---

## 4. Methodological Comparison: Paradigm 1 vs. Paradigm 2

| Dimension | Paradigm 1: Unsupervised RPCA (IALM) | Paradigm 2: Learned Healthy Operator ($W^*$) |
| :--- | :--- | :--- |
| **Supervision Level** | **Fully Unsupervised** (Zero labels) | **Semi-Supervised** (Healthy cohort only) |
| **Data Requirement** | Unlabelled mixed cohort ($N$ normal + diseased scans together) | Reference set of confirmed healthy scans ($X_H$) |
| **Mathematical Basis** | Low-rank nuclear norm + sparse $L_1$ norm decomposition ($X = L + S$) | Tikhonov-regularized spatial mapping $W^* = (X_H^T X_H + \alpha I)^{-1} X_H^T X_H$ |
| **Anomaly Signal** | Sparse matrix $S$ | Annihilation residual $r = x_{\text{query}}(I - W^*)$ |
| **Operator Role** | Joint cohort matrix factorization | Pre-computed static linear operator applied to single test scans |
| **Test-Time Complexity** | Iterative SVD ($\approx 30$–$50$ iterations) | Instant single matrix-vector multiplication ($\mathcal{O}(P'^2)$ FLOPs) |
| **Best Clinical Setup** | *De-novo* anomaly discovery on uncurated clinical archives | High-throughput deployment with pre-calibrated anatomical atlases |
