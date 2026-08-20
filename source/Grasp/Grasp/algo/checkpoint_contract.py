from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TEACHER_OBSERVATION_DIM = 119
TEACHER_ACTION_DIM = 13

TEACHER_PLAY_KEYS = (
    "actor_architecture_state_dict",
    "actor_distribution_state_dict",
    "obs_spec",
    "action_dim",
)

TEACHER_EXPERT_KEYS = (
    *TEACHER_PLAY_KEYS,
    "critic_architecture_state_dict",
)

TEACHER_RESUME_KEYS = (
    *TEACHER_EXPERT_KEYS,
    "optimizer_state_dict",
    "update",
)

STUDENT_PLAY_KEYS = (
    "actor_architecture_state_dict",
    "actor_distribution_state_dict",
    "prop_encoder_state_dict",
    "obs_spec",
    "student_actor_obs_dim",
    "action_dim",
)

STUDENT_RESUME_KEYS = (
    *STUDENT_PLAY_KEYS,
    "critic_architecture_state_dict",
    "optimizer_state_dict",
    "update",
)

OBS_SPEC_FIELDS = (
    "base_dim",
    "aff_vec_dim",
    "teacher_dim",
    "history_dim",
    "history_len",
    "student_total_dim",
    "prop_latent_dim",
)


def require_checkpoint_keys(
    checkpoint: Mapping[str, Any],
    required_keys: Sequence[str],
    *,
    role: str,
    path: str,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"{role} checkpoint must be a mapping: {path}; "
            f"got {type(checkpoint).__name__}"
        )

    missing_keys = sorted(set(required_keys) - set(checkpoint.keys()))
    if missing_keys:
        raise RuntimeError(
            f"{role} checkpoint is missing required keys: {missing_keys}; path={path}"
        )


def _require_mapping(value: Any, *, field: str, role: str, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"{role} checkpoint field '{field}' must be a mapping: {path}; "
            f"got {type(value).__name__}"
        )
    return value


def _require_int(value: Any, *, field: str, role: str, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"{role} checkpoint field '{field}' must be an integer: {path}; "
            f"got {type(value).__name__}"
        )
    return value


def validate_teacher_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_keys: Sequence[str],
    expected_teacher_dim: int,
    expected_action_dim: int,
    path: str,
) -> None:
    role = "Teacher"
    require_checkpoint_keys(checkpoint, required_keys, role=role, path=path)

    if expected_teacher_dim != TEACHER_OBSERVATION_DIM:
        raise RuntimeError(
            "Teacher environment observation dimension must be "
            f"{TEACHER_OBSERVATION_DIM}; got {expected_teacher_dim}"
        )
    if expected_action_dim != TEACHER_ACTION_DIM:
        raise RuntimeError(
            f"Teacher environment action dimension must be {TEACHER_ACTION_DIM}; "
            f"got {expected_action_dim}"
        )
    if "student_actor_obs_dim" in checkpoint:
        raise RuntimeError(
            f"Teacher checkpoint contains Student-only metadata: {path}"
        )

    obs_spec = _require_mapping(
        checkpoint["obs_spec"], field="obs_spec", role=role, path=path
    )
    if "teacher_dim" not in obs_spec:
        raise RuntimeError(
            f"Teacher checkpoint obs_spec is missing 'teacher_dim': {path}"
        )

    checkpoint_teacher_dim = _require_int(
        obs_spec["teacher_dim"],
        field="obs_spec.teacher_dim",
        role=role,
        path=path,
    )
    checkpoint_action_dim = _require_int(
        checkpoint["action_dim"],
        field="action_dim",
        role=role,
        path=path,
    )

    if checkpoint_teacher_dim != TEACHER_OBSERVATION_DIM:
        raise RuntimeError(
            "Teacher checkpoint observation dimension mismatch: "
            f"checkpoint={checkpoint_teacher_dim}, required={TEACHER_OBSERVATION_DIM}, "
            f"path={path}"
        )
    if checkpoint_action_dim != TEACHER_ACTION_DIM:
        raise RuntimeError(
            "Teacher checkpoint action dimension mismatch: "
            f"checkpoint={checkpoint_action_dim}, required={TEACHER_ACTION_DIM}, "
            f"path={path}"
        )


def validate_student_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_keys: Sequence[str],
    expected_obs_spec: Any,
    expected_student_actor_obs_dim: int,
    expected_action_dim: int,
    path: str,
) -> None:
    role = "Student"
    require_checkpoint_keys(checkpoint, required_keys, role=role, path=path)

    obs_spec = _require_mapping(
        checkpoint["obs_spec"], field="obs_spec", role=role, path=path
    )
    missing_fields = sorted(set(OBS_SPEC_FIELDS) - set(obs_spec.keys()))
    if missing_fields:
        raise RuntimeError(
            f"Student checkpoint obs_spec is missing fields: {missing_fields}; path={path}"
        )

    for field in OBS_SPEC_FIELDS:
        checkpoint_value = _require_int(
            obs_spec[field],
            field=f"obs_spec.{field}",
            role=role,
            path=path,
        )
        expected_value = getattr(expected_obs_spec, field)
        if checkpoint_value != expected_value:
            raise RuntimeError(
                f"Student checkpoint obs_spec.{field} mismatch: "
                f"checkpoint={checkpoint_value}, environment={expected_value}, path={path}"
            )

    checkpoint_student_dim = _require_int(
        checkpoint["student_actor_obs_dim"],
        field="student_actor_obs_dim",
        role=role,
        path=path,
    )
    checkpoint_action_dim = _require_int(
        checkpoint["action_dim"],
        field="action_dim",
        role=role,
        path=path,
    )

    if checkpoint_student_dim != expected_student_actor_obs_dim:
        raise RuntimeError(
            "Student checkpoint actor observation dimension mismatch: "
            f"checkpoint={checkpoint_student_dim}, "
            f"environment={expected_student_actor_obs_dim}, path={path}"
        )
    if checkpoint_action_dim != expected_action_dim:
        raise RuntimeError(
            "Student checkpoint action dimension mismatch: "
            f"checkpoint={checkpoint_action_dim}, environment={expected_action_dim}, "
            f"path={path}"
        )


def get_resume_start_update(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
    path: str,
) -> int:
    require_checkpoint_keys(checkpoint, ("update",), role=role, path=path)
    update = _require_int(
        checkpoint["update"], field="update", role=role, path=path
    )
    if update < 0:
        raise RuntimeError(
            f"{role} checkpoint update must be non-negative: update={update}, path={path}"
        )
    return update + 1