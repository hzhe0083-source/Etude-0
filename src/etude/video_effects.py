"""Continuous video-effect bottleneck and masked multi-time prediction.

Precomputed, frozen visual features are inputs. There is no visual backbone,
detector, action label, task requirement, view-pair requirement or download here.
Reconstruction and a small bottleneck alone do not establish view invariance.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import _pair_features


def _observed(features: Tensor, valid: Tensor, name: str) -> Tensor:
    if (not isinstance(features, Tensor) or features.ndim != 4 or not features.is_floating_point()
            or min(features.shape) <= 0):
        raise ValueError(f"{name} must be nonempty floating [B,T,N,D]")
    if (not isinstance(valid, Tensor) or valid.shape != features.shape or valid.dtype != torch.bool
            or valid.device != features.device):
        raise ValueError(f"{name} validity must be data-owned bool with the same shape/device")
    if not torch.isfinite(features[valid]).all():
        raise ValueError(f"valid {name} features must be finite")
    return torch.where(valid, features, 0)


def _times(times: Tensor, count: int, reference: Tensor, *, future=False) -> Tensor:
    if (not isinstance(times, Tensor) or times.shape != (count,) or times.dtype == torch.bool
            or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any()):
        raise ValueError("times must be finite, strictly increasing [T] values")
    if future and (count < 2 or (times <= 0).any()):
        raise ValueError("at least two positive future query offsets are required: intermediate and terminal")
    return times.to(device=reference.device, dtype=reference.dtype)


def _positions(feature_kind: str, coordinates: Tensor | None, reference: Tensor) -> Tensor | None:
    """Patch locations are observed inputs; entity table indices are never locations."""
    if feature_kind == "tracked_entities":
        if coordinates is not None:
            raise ValueError("tracked entities must not use patch coordinates or entity IDs as positions")
        return None
    if feature_kind != "patches":
        raise ValueError("feature_kind must be patches or tracked_entities")
    if (not isinstance(coordinates, Tensor) or coordinates.shape != (reference.shape[2], 2)
            or not coordinates.is_floating_point() or not torch.isfinite(coordinates).all()
            or (coordinates.abs() >= 1).any() or coordinates.unique(dim=0).shape[0] != reference.shape[2]):
        raise ValueError("patches require unique finite normalized xy patch centers [N,2] in (-1,1)")
    return coordinates.to(reference)


class VideoEffectEncoder(nn.Module):
    """Encode a process-and-result window into fixed K continuous tokens.

    Patches bind content to explicit image coordinates before pooling. Stable
    entity trajectories are encoded before set pooling, without ID features.
    ``feature_valid`` describes visible input data,
    not a model confidence prediction. Tokens are bounded by tanh and optional
    clipped Gaussian perturbation; this is not a formal information-rate bound.
    """

    # Capacity candidate, not an experimentally established optimum. Unit-test
    # configurations pass their small dimensions explicitly.
    def __init__(self, feature_dim: int, latent_dim: int = 768, num_tokens: int = 64,
                 hidden_dim: int = 512, noise_std: float = .05):
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (feature_dim, latent_dim, num_tokens, hidden_dim)):
            raise ValueError("effect encoder dimensions and token count must be positive integers")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.feature_dim, self.latent_dim = feature_dim, latent_dim
        self.num_tokens, self.hidden_dim, self.noise_std = num_tokens, hidden_dim, float(noise_std)
        self.features = nn.Linear(2 * feature_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.position = nn.Linear(2, hidden_dim, bias=False)
        self.mix = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.trajectory = nn.GRUCell(hidden_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.keys = nn.Linear(hidden_dim, hidden_dim)
        self.values = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, features: Tensor, feature_valid: Tensor, frame_times: Tensor,
                noise: bool = True, *, feature_kind: str,
                patch_coordinates: Tensor | None = None) -> Tensor:
        clean = _observed(features, feature_valid, "window")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("window feature dimension differs from encoder")
        present = feature_valid.any(-1).flatten(1)
        if not present.any(-1).all():
            raise ValueError("each process window requires observed features")
        times = _times(frame_times, features.shape[1], features)
        relative = torch.log1p(times - times[0])
        hidden = self.features(torch.cat((clean, feature_valid.to(clean.dtype)), -1))
        hidden = hidden + self.time(relative[:, None])[None, :, None]
        coordinates = _positions(feature_kind, patch_coordinates, features)
        if coordinates is not None:
            hidden = hidden + self.position(coordinates)[None, None]
        # A nonlinear content-position-time interaction must precede pooling.
        hidden = self.mix(hidden)
        if feature_kind == "tracked_entities":
            batch, length, entities, width = hidden.shape
            state = hidden.new_zeros(batch * entities, width)
            trajectory = []
            for step in range(length):
                candidate = self.trajectory(hidden[:, step].reshape(batch * entities, width), state)
                observed = feature_valid[:, step].any(-1).reshape(-1, 1)
                state = torch.where(observed, candidate, state)
                trajectory.append(state.reshape(batch, entities, width))
            hidden = torch.stack(trajectory, dim=1)
        hidden = hidden.flatten(1, 2)
        scores = torch.einsum("kd,bsd->bks", self.queries, self.keys(hidden)) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~present[:, None], torch.finfo(scores.dtype).min)
        pooled = torch.einsum("bks,bsd->bkd", scores.softmax(-1), self.values(hidden))
        tokens = self.output(pooled).tanh()
        if self.training and noise and self.noise_std:
            tokens = (tokens + self.noise_std * torch.randn_like(tokens)).clamp(-1, 1)
        return tokens

    def encode_demo(self, features: Tensor, feature_valid: Tensor, frame_times: Tensor,
                    window_frames: int = 5, *, feature_kind: str,
                    patch_coordinates: Tensor | None = None) -> Tensor:
        """Ordered nonoverlapping windows; preserve every real tail frame.

        A short tail repeats the final numeric value but marks padded entries
        invalid. No random perturbation is used, even in training mode. Autograd
        is left to the caller: inference should freeze B and use no_grad().
        ``window_frames`` must match the registered pretraining window length.
        """
        _observed(features, feature_valid, "demonstration")
        if type(window_frames) is not int or window_frames < 3:
            raise ValueError("window_frames must leave room for a past and two future frames")
        times = _times(frame_times, features.shape[1], features)
        step = times[-1] - times[-2] if len(times) > 1 else times.new_tensor(1.)
        output = []
        for start in range(0, features.shape[1], window_frames):
            window = features[:, start:start + window_frames]
            valid = feature_valid[:, start:start + window_frames]
            window_times = times[start:start + window_frames]
            padding = window_frames - window.shape[1]
            if padding:
                window = torch.cat((window, window[:, -1:].expand(-1, padding, -1, -1)), 1)
                valid = torch.cat((valid, torch.zeros_like(valid[:, -1:]).expand(-1, padding, -1, -1)), 1)
                future_times = window_times[-1] + step * torch.arange(1, padding + 1, device=times.device, dtype=times.dtype)
                window_times = torch.cat((window_times, future_times))
            output.append(self(window, valid, window_times, noise=False,
                               feature_kind=feature_kind, patch_coordinates=patch_coordinates))
        return torch.cat(output, 1)


class EffectFeaturePredictor(nn.Module):
    """Predict future features using past appearance and only bottleneck z.

    ``query_times`` are positive offsets from the final past observation. Pass
    ``past_times`` for irregularly sampled histories; otherwise unit spacing is
    assumed. A target/clean-future/process argument deliberately does not exist.
    This task-conditioned predictor is not the independent physical predictor F.
    """

    def __init__(self, feature_dim: int, latent_dim: int = 768, hidden_dim: int = 512,
                 geometry_dim: int = 0, relation_dim: int = 0, event_dim: int = 0):
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (feature_dim, latent_dim, hidden_dim)):
            raise ValueError("predictor feature/latent/hidden dimensions must be positive integers")
        if any(type(size) is not int or size < 0 for size in (geometry_dim, relation_dim, event_dim)):
            raise ValueError("optional effect vocabulary dimensions must be nonnegative integers")
        self.feature_dim, self.latent_dim, self.hidden_dim = feature_dim, latent_dim, hidden_dim
        self.geometry_dim, self.relation_dim, self.event_dim = geometry_dim, relation_dim, event_dim
        self.history = nn.GRU(2 * feature_dim + 1, hidden_dim, batch_first=True)
        self.position = nn.Linear(2, hidden_dim, bias=False)
        self.token_keys = nn.Linear(latent_dim, hidden_dim)
        self.token_values = nn.Linear(latent_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU())
        self.feature_head = nn.Linear(hidden_dim, feature_dim)
        self.geometry_head = nn.Linear(hidden_dim, geometry_dim) if geometry_dim else None
        self.relation_head = nn.Linear(2 * hidden_dim, relation_dim) if relation_dim else None
        self.event_head = nn.Linear(2 * hidden_dim, event_dim) if event_dim else None

    def forward(self, past: Tensor, past_valid: Tensor, tokens: Tensor, query_times: Tensor,
                *, past_times: Tensor | None = None,
                effect_fields: tuple[str, ...] | None = None, feature_kind: str,
                patch_coordinates: Tensor | None = None) -> dict[str, Tensor]:
        clean = _observed(past, past_valid, "past")
        batch, length, entities, width = past.shape
        if width != self.feature_dim:
            raise ValueError("past feature dimension differs from predictor")
        requested = {"geometry", "relations", "events"} if effect_fields is None else set(effect_fields)
        if requested - {"geometry", "relations", "events"}:
            raise ValueError("effect_fields may only request geometry, relations and events")
        if (tokens.ndim != 3 or tokens.shape[0] != batch or tokens.shape[-1] != self.latent_dim
                or not tokens.shape[1] or not tokens.is_floating_point() or not torch.isfinite(tokens).all()
                or tokens.device != past.device):
            raise ValueError("tokens must be finite [B,K,latent_dim] on the past-feature device")
        if past_times is None:
            past_times = torch.arange(length, device=past.device, dtype=past.dtype)
        times = _times(past_times, length, past)
        relative = times - times[-1]
        history_time = relative.sign() * torch.log1p(relative.abs())
        history_time = history_time[None, :, None, None].expand(batch, -1, entities, -1)
        inputs = torch.cat((clean, past_valid.to(clean.dtype), history_time), -1)
        inputs = inputs.transpose(1, 2).reshape(batch * entities, length, -1)
        _, encoded = self.history(inputs)
        state = encoded[-1].reshape(batch, entities, self.hidden_dim)
        coordinates = _positions(feature_kind, patch_coordinates, past)
        if coordinates is not None:
            state = state + self.position(coordinates)[None]
        queries = _times(query_times, query_times.numel(), past, future=True)
        state = state[:, None] + self.time(torch.log1p(queries[:, None]))[None, :, None]
        scores = torch.einsum("bqnd,bkd->bqnk", state, self.token_keys(tokens)) / math.sqrt(self.hidden_dim)
        effect = torch.einsum("bqnk,bkd->bqnd", scores.softmax(-1), self.token_values(tokens))
        hidden = self.fuse(torch.cat((state, effect), -1))
        # Retain each channel's last observed value; a missing final frame must
        # not erase appearance already available in actual past observations.
        index = torch.arange(length, device=past.device)[None, :, None, None].expand_as(past_valid)
        last = torch.where(past_valid, index, -1).amax(1)
        appearance = clean.gather(1, last.clamp_min(0)[:, None]).squeeze(1)
        appearance = torch.where(last >= 0, appearance, 0)
        # Unpaired patch-feature videos can have thousands of slots. Without
        # pair labels, even constructing a pair tensor would waste O(N^2) memory.
        need_relations = "relations" in requested and self.relation_head is not None
        need_events = "events" in requested and self.event_head is not None
        pairs = _pair_features(hidden) if need_relations or need_events else None
        empty_pairs = hidden.new_empty(batch, len(queries), entities, entities, 0)
        return {
            "features": appearance[:, None] + self.feature_head(hidden),
            "geometry": self.geometry_head(hidden) if "geometry" in requested and self.geometry_head is not None else hidden[..., :0],
            "relations": self.relation_head(pairs) if need_relations else empty_pairs,
            "events": self.event_head(pairs) if need_events else empty_pairs,
        }


def effect_pretraining_loss(predictions: Mapping[str, Tensor], features_target: Tensor,
                            feature_valid: Tensor, effect_targets: Mapping[str, Tensor] | None,
                            effect_valid: Mapping[str, Tensor] | None, tokens: Tensor,
                            weights: Mapping[str, float] | None = None) -> dict[str, Tensor]:
    """Masked frozen-target losses; capacity never trains on unlabeled data.

    Label masks are supplied by the dataset, never the model. Unknown entries
    may hold NaNs: replace them before any arithmetic. Empty supervision returns
    a detached zero total so the caller can skip backward and optimizer.step().
    Token energy is merely a small regularizer, not an information-rate claim.
    """
    if set(predictions) != {"features", "geometry", "relations", "events"}:
        raise ValueError("predictions must contain features, geometry, relations and events")
    scales = {"features": 1., "geometry": 1., "relations": 1., "events": 1., "capacity": .001}
    if weights is not None:
        if set(weights) - set(scales):
            raise ValueError("unknown pretraining loss weight")
        scales.update(weights)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in scales.values()):
        raise ValueError("pretraining weights must be finite and nonnegative")
    effect_targets, effect_valid = dict(effect_targets or {}), dict(effect_valid or {})
    if set(effect_targets) != set(effect_valid) or set(effect_targets) - {"geometry", "relations", "events"}:
        raise ValueError("optional effect targets and data validity must have the same known keys")
    targets = {"features": features_target, **effect_targets}
    validity = {"features": feature_valid, **effect_valid}
    reference = predictions["features"]
    if (tokens.ndim != 3 or not tokens.is_floating_point() or not torch.isfinite(tokens).all()
            or tokens.device != reference.device or tokens.shape[0] != reference.shape[0] or not tokens.numel()):
        raise ValueError("capacity regularization requires finite matching batched tokens")
    zero = reference.new_zeros((), dtype=torch.float32)
    result = {name: zero for name in scales}
    valid_count = 0
    for name, target in targets.items():
        prediction, valid = predictions[name], validity[name]
        if (not target.is_floating_point() or target.shape != prediction.shape
                or target.device != prediction.device or valid.dtype != torch.bool
                or valid.shape != target.shape or valid.device != target.device):
            raise ValueError(f"{name} target/validity must match the prediction shape/device")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
            raise ValueError(f"valid {name} targets and predictions must be finite")
        if name in {"relations", "events"} and ((target[valid] != 0) & (target[valid] != 1)).any():
            raise ValueError("observed relationships and events must be binary where valid")
        safe = torch.where(valid, target.detach(), 0).float()
        raw = (F.binary_cross_entropy_with_logits(prediction.float(), safe, reduction="none")
               if name in {"relations", "events"} else (prediction.float() - safe).square())
        result[name] = torch.where(valid, raw, 0).sum() / valid.sum().clamp_min(1)
        if scales[name] > 0:
            valid_count += int(valid.sum())
    result["valid_count"] = zero.new_tensor(valid_count)
    if not valid_count:
        result["capacity"], result["total"] = zero, zero
        return result
    result["capacity"] = tokens.float().square().mean()
    result["total"] = sum((scales[name] * result[name] for name in scales), zero)
    return result
