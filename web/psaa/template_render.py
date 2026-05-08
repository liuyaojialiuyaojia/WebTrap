from __future__ import annotations

import re
from typing import Any, Mapping

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Use sentinels that are vanishingly unlikely to appear in prompts.
_ESC_L = "\u0000LBRACE\u0000"
_ESC_R = "\u0000RBRACE\u0000"


def render_template(template: str, variables: Mapping[str, Any]) -> str:
    """Render a prompt template with {placeholders}.

    Supports a subset of Python's `.format()` escaping rules:
      - `{{` becomes a literal `{`
      - `}}` becomes a literal `}`

    Unlike `.format()`, this renderer only substitutes `{ident}` placeholders,
    so JSON examples like `{"tag": "yes"}` are left intact (or can be written
    as `{{"tag": "yes"}}` to avoid accidental format semantics elsewhere).
    """
    if not isinstance(template, str):
        raise TypeError(f"template must be str, got {type(template).__name__}")

    # Preserve escaped braces first.
    material = template.replace("{{", _ESC_L).replace("}}", _ESC_R)

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Missing template variable: {key}")
        value = variables[key]
        if value is None:
            return ""
        return str(value)

    rendered = _PLACEHOLDER_RE.sub(_sub, material)
    return rendered.replace(_ESC_L, "{").replace(_ESC_R, "}")

