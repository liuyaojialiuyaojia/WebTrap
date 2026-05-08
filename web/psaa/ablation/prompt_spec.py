from __future__ import annotations

from pathlib import Path

from psaa.prompt_spec import PsaaPromptSpec, load_psaa_prompt_spec


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PSAA_PROMPT_SPEC_VERSION = "base"
VALID_PSAA_PROMPT_SPEC_VERSIONS = ("base", "enhanced")
_DEFAULT_PSAA_PROMPT_SPEC_PATHS = {
    "base": REPO_ROOT / "psaa" / "psaa_v1.yaml",
    "enhanced": REPO_ROOT / "psaa" / "psaa_v2.yaml",
}


def normalize_prompt_spec_version(value: str | None) -> str:
    version = str(value or DEFAULT_PSAA_PROMPT_SPEC_VERSION).strip().lower()
    if version not in VALID_PSAA_PROMPT_SPEC_VERSIONS:
        allowed = ", ".join(VALID_PSAA_PROMPT_SPEC_VERSIONS)
        raise ValueError(
            f"Unsupported PSAA prompt spec version: {version!r}. Allowed values: {allowed}."
        )
    return version


def resolve_ablation_prompt_spec_path(
    prompt_spec: Path | None,
    prompt_spec_version: str | None,
) -> Path:
    if prompt_spec is not None:
        return prompt_spec
    version = normalize_prompt_spec_version(prompt_spec_version)
    return _DEFAULT_PSAA_PROMPT_SPEC_PATHS[version]


def load_ablation_prompt_spec(path: Path) -> PsaaPromptSpec:
    return load_psaa_prompt_spec(path)
