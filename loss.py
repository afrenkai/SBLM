import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask


class SIGRegLoss(nn.Module):
    """
    Sliced Isotropic Gaussian Regularization (Epps-Pulley 1D characteristic function test).
    Forces the embedding distribution toward N(0, I_d) across random 1D projections.
    """
    def __init__(
        self,
        d_model: int,
        num_projections: int = 256,
        num_quad_points: int = 17,
        omega_min: float = 0.2,
        omega_max: float = 4.0,
        vocab_sample_size: int = 2048,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.vocab_sample_size = vocab_sample_size

        # 17-point Gauss-Legendre quadrature nodes & weights mapped to [omega_min, omega_max]
        nodes, weights = np.polynomial.legendre.leggauss(num_quad_points)
        half_span = 0.5 * (omega_max - omega_min)
        mid = 0.5 * (omega_max + omega_min)
        omegas = torch.tensor(mid + half_span * nodes, dtype=torch.float32)          # (K,)
        quad_weights = torch.tensor(half_span * weights, dtype=torch.float32)       # (K,)

        # Analytic N(0, 1) characteristic function & Gaussian integration kernel e^(-w^2 / 2)
        target_cf = torch.exp(-0.5 * omegas.pow(2))                                 # (K,)
        weighted_quad = quad_weights * target_cf                                    # (K,)

        self.register_buffer("omegas", omegas, persistent=False)
        self.register_buffer("target_cf", target_cf, persistent=False)
        self.register_buffer("weighted_quad", weighted_quad, persistent=False)

    def forward(self, batch_embs: torch.Tensor, vocab_weight: torch.Tensor) -> torch.Tensor:
        """
        batch_embs:   (N_batch, D) flattened target embeddings E(w)
        vocab_weight: (V, D) full embedding table E.weight
        """
        # Mix batch embeddings with uniform vocabulary rows to prevent Zipfian rare-token drift
        V, D = vocab_weight.shape
        rand_idx = torch.randint(0, V, (self.vocab_sample_size,), device=vocab_weight.device)
        z = torch.cat([batch_embs.float(), vocab_weight[rand_idx].float()], dim=0)  # (N, D)

        # Sample M random unit directions on S^(D-1)
        u = torch.randn((D, self.num_projections), device=z.device, dtype=torch.float32)
        u = F.normalize(u, dim=0)

        # 1D projections: (N, M)
        proj = z @ u

        # Evaluate empirical characteristic function across K quadrature frequencies: (N, M, K)
        angles = proj.unsqueeze(-1) * self.omegas
        emp_cos = torch.cos(angles).mean(dim=0)  # (M, K)
        emp_sin = torch.sin(angles).mean(dim=0)  # (M, K)

        # |phi_emp(w) - e^(-w^2/2)|^2 = (Re - e^(-w^2/2))^2 + Im^2
        sq_diff = (emp_cos - self.target_cf).pow(2) + emp_sin.pow(2)  # (M, K)

        # Integrate over omega and average over M projections
        return (sq_diff * self.weighted_quad).sum(dim=-1).mean()


class BrownianBridgeCriterion(nn.Module):
    def __init__(
        self,
        sigma: float = 0.3,
        w_enc_ce: float = 0.1,
        w_dir: float = 0.1,
        w_sigreg: float = 0.1,
        w_cons: float = 0.05,
        cons_interval: int = 4,
        cons_delta: float = 0.03,
    ):
        super().__init__()
        self.sigma = sigma
        self.w_enc_ce = w_enc_ce
        self.w_dir = w_dir
        self.w_sigreg = w_sigreg
        self.w_cons = w_cons
        self.cons_interval = cons_interval
        self.cons_delta = cons_delta
        self.sigreg = None  # Lazily initialized with d_model on first call

    def sample_bridge(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Closed-form Brownian bridge marginal: X_t = (1-t)X_0 + t X_T + sigma * sqrt(t(1-t)) * eps."""
        t = tau.float().unsqueeze(-1)
        std = self.sigma * torch.sqrt(torch.clamp(t * (1.0 - t), min=0.0))
        x_t = (1.0 - t) * x_0 + t * x_T + std * eps
        return x_t.to(out_dtype)

    def forward(
        self,
        model: nn.Module,
        raw_model: nn.Module,  # Unwrapped BrownianDDT (in case model is DDP/compiled)
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tau: torch.Tensor,
        rope_pos: torch.Tensor,
        block_mask: BlockMask,
        step: int,
        x_0_override: torch.Tensor | None = None,  # Used in Stage 2 (k=1) for coupled IMF pairs
    ) -> tuple[torch.Tensor, dict[str, float]]:
        E_weight = raw_model.tok_emb.weight
        D = E_weight.shape[1]
        if self.sigreg is None:
            self.sigreg = SIGRegLoss(d_model=D).to(E_weight.device)

        # 1. Construct Brownian bridge state X_t in float32, cast to model dtype (bfloat16)
        x_T = F.embedding(input_ids, E_weight).float()
        x_0 = torch.randn_like(x_T) if x_0_override is None else x_0_override.float()
        eps = torch.randn_like(x_T)
        x_t = self.sample_bridge(x_0, x_T, tau, eps, out_dtype=E_weight.dtype)

        # 2. Forward pass through Encoder + DDT Decoder
        logits, z_t, enc_ce_loss = model(x_t, tau, rope_pos, block_mask, labels)

        # 3. Main Decoder Cross-Entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 4. Singularity-Free Velocity Direction Loss: 1 - cos(x_hat - X_t, X_T - X_t)
        valid_mask = labels != -100
        probs = torch.softmax(logits.float(), dim=-1)
        x_hat = probs @ E_weight.float()  # Clamped embedding on vocabulary convex hull
        x_t_f32 = x_t.float()

        pred_vec = x_hat[valid_mask] - x_t_f32[valid_mask]
        target_vec = x_T[valid_mask] - x_t_f32[valid_mask]
        dir_loss = (1.0 - F.cosine_similarity(pred_vec, target_vec, dim=-1, eps=1e-6)).mean()

        # 5. SIGReg on target embeddings + random vocabulary subset
        sigreg_loss = self.sigreg(x_T.view(-1, D), E_weight)

        # 6. Adjacent-Step Encoder Consistency (runs every `cons_interval` steps to reuse block_mask)
        if self.w_cons > 0.0 and (step % self.cons_interval == 0):
            delta = torch.empty_like(tau).uniform_(-self.cons_delta, self.cons_delta)
            tau_prime = torch.clamp(tau + delta, 0.0, 1.0)
            # Keep clean prefix tokens pinned at tau = 1.0
            tau_prime = torch.where(tau >= 1.0, torch.ones_like(tau_prime), tau_prime)

            x_t_prime = self.sample_bridge(x_0, x_T, tau_prime, eps, out_dtype=E_weight.dtype)
            with torch.no_grad():
                z_t_prime = raw_model.encode(x_t_prime, tau_prime, rope_pos, block_mask)

            cons_loss = (
                1.0 - F.cosine_similarity(z_t.float()[valid_mask], z_t_prime.float()[valid_mask], dim=-1, eps=1e-6)
            ).mean()
            scaled_cons = (self.w_cons * self.cons_interval) * cons_loss
        else:
            cons_loss = torch.zeros((), device=ce_loss.device)
            scaled_cons = cons_loss

        total_loss = (
            ce_loss
            + self.w_enc_ce * enc_ce_loss
            + self.w_dir * dir_loss
            + self.w_sigreg * sigreg_loss
            + scaled_cons
        )

        with torch.no_grad():
            emb_std = E_weight.float().std(dim=0).mean().item()

        metrics = {
            "loss": total_loss.detach().item(),
            "ce": ce_loss.detach().item(),
            "enc_ce": enc_ce_loss.detach().item(),
            "dir": dir_loss.detach().item(),
            "sigreg": sigreg_loss.detach().item(),
            "cons": cons_loss.detach().item(),
            "emb_std": emb_std,
        }
        return total_loss, metrics
