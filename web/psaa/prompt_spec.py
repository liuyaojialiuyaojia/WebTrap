from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AttackerStageTemplate:
    first: str
    rewrite: str


@dataclass(frozen=True)
class AttackerPromptTemplates:
    lure: AttackerStageTemplate
    inertia: AttackerStageTemplate
    payload: AttackerStageTemplate


@dataclass(frozen=True)
class CheckPromptTemplates:
    lure: str
    inertia: str
    payload: str


@dataclass(frozen=True)
class LLMConfig:
    model: str
    system_message: str


@dataclass(frozen=True)
class PsaaPromptSpec:
    version: str
    retry_num: int
    attacker: LLMConfig
    attacker_templates: AttackerPromptTemplates
    checker: LLMConfig
    check_templates: CheckPromptTemplates


def _require_str(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PSAA prompt spec {key} must be a non-empty string.")
    return value.strip()


def _require_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"PSAA prompt spec {key} must be an int.")
    return int(value)


def _require_mapping(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = obj.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"PSAA prompt spec {key} must be an object/mapping.")
    return value


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load PSAA prompt templates. Install PyYAML (pip install pyyaml)."
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_psaa_prompt_spec(path: Path) -> PsaaPromptSpec:
    raw = _load_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError("PSAA prompt spec must be a YAML mapping/object.")

    version = _require_str(raw, "version")
    retry_num = _require_int(raw, "retry_num")
    if retry_num <= 0:
        raise ValueError("PSAA prompt spec retry_num must be positive.")

    attacker_raw = _require_mapping(raw, "attacker_llm")
    attacker_model = _require_str(attacker_raw, "model")
    attacker_system = _require_str(attacker_raw, "system_message")
    attacker_tpl_raw = _require_mapping(attacker_raw, "user_prompt_template")

    def _load_attacker_stage(stage: str) -> AttackerStageTemplate:
        stage_raw = _require_mapping(attacker_tpl_raw, stage)
        return AttackerStageTemplate(
            first=_require_str(stage_raw, "first"),
            rewrite=_require_str(stage_raw, "rewrite"),
        )

    attacker_templates = AttackerPromptTemplates(
        lure=_load_attacker_stage("lure"),
        inertia=_load_attacker_stage("inertia"),
        payload=_load_attacker_stage("payload"),
    )

    checker_raw = _require_mapping(raw, "check_llm")
    checker_model = _require_str(checker_raw, "model")
    checker_system = _require_str(checker_raw, "system_message")
    checker_tpl_raw = _require_mapping(checker_raw, "user_prompt_template")

    check_templates = CheckPromptTemplates(
        lure=_require_str(checker_tpl_raw, "lure"),
        inertia=_require_str(checker_tpl_raw, "inertia"),
        payload=_require_str(checker_tpl_raw, "payload"),
    )

    return PsaaPromptSpec(
        version=version,
        retry_num=retry_num,
        attacker=LLMConfig(model=attacker_model, system_message=attacker_system),
        attacker_templates=attacker_templates,
        checker=LLMConfig(model=checker_model, system_message=checker_system),
        check_templates=check_templates,
    )

