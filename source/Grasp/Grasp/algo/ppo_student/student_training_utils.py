from __future__ import annotations

import torch


def compute_student_actor_obs_dim(
    history_dim: int,
    prop_latent_dim: int,
    affordance_vec_dim: int,
) -> int:
    return history_dim + prop_latent_dim + affordance_vec_dim


def copy_compatible_teacher_actor_layers(
    teacher_mlp,
    student_mlp,
) -> tuple[str, ...]:
    """Copy all compatible Teacher MLP tensors into the Student MLP."""
    teacher_state = teacher_mlp.state_dict()
    student_state = student_mlp.state_dict()
    mismatched = []

    for name, student_tensor in student_state.items():
        teacher_tensor = teacher_state[name]
        if teacher_tensor.shape == student_tensor.shape:
            student_state[name] = teacher_tensor.detach().clone()
        else:
            mismatched.append(name)

    expected_mismatch = ("architecture.0.weight",)
    if tuple(mismatched) != expected_mismatch:
        raise ValueError(
            "Unexpected Teacher/Student actor shape mismatch: "
            f"expected={expected_mismatch}, actual={tuple(mismatched)}"
        )

    student_mlp.load_state_dict(student_state)
    return tuple(mismatched)

def build_student_actor_observation(
    total_obs: torch.Tensor,
    prop_latent_encoder,
    history_len: int,
    history_dim: int,
    student_affordance: torch.Tensor,
) -> torch.Tensor:
    """Build the 88-D Student Actor input without teacher privileged tail."""
    history_flat_dim = history_len * history_dim
    history = total_obs[:, :history_flat_dim]
    latest_history = history[:, -history_dim:]
    latent = prop_latent_encoder(history)
    return torch.cat(
        [latest_history, latent, student_affordance],
        dim=-1,
    )
