#!/usr/bin/env python3
"""Data structures for Stage 1 node construction requests."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Sequence, Tuple
from types import MappingProxyType


@dataclasses.dataclass(frozen=True)
class SiblingInfo:
    """Summary information for a sibling node."""

    title: str
    summary: str | None = None


@dataclasses.dataclass(frozen=True)
class PageRequest:
    """Context required to generate one page."""

    page_id: str
    page_index: int
    breadcrumb: Sequence[int]
    gen_mode: str = "direct"
    title_hint: str | None = None
    title_path: Sequence[str] = dataclasses.field(default_factory=tuple)
    parent_summary: str | None = None
    siblings: Sequence[SiblingInfo] = dataclasses.field(default_factory=tuple)
    batch_size: int = 1
    seed: int | None = None

    @property
    def ancestor_titles(self) -> Sequence[str]:
        """Backward-compatible alias for older field names."""

        return self.title_path

    @property
    def father(self) -> str | None:
        """Backward-compatible alias for older field names."""

        return self.parent_summary


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    """Action definition with a primitive concept and optional properties."""

    concept: str
    props: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        props = dict(self.props or {})
        object.__setattr__(self, "props", MappingProxyType(props))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"concept": self.concept}
        if self.props:
            payload["props"] = dict(self.props)
        return payload


@dataclasses.dataclass(frozen=True)
class PageDraft:
    """Raw page data returned by the LLM before page IDs are attached."""

    title: str
    body: str
    action: Tuple[ActionSpec, ...]
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "body": self.body,
            "action": [item.to_dict() for item in self.action],
            "summary": self.summary,
        }


@dataclasses.dataclass(frozen=True)
class PageMetadata:
    """Page definition emitted by Stage 1."""

    page_id: str
    page_index: int
    breadcrumb: str
    title: str
    summary: str
    body: str
    action: Tuple[ActionSpec, ...]
    action_page: Tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "page_id": self.page_id,
            "page_index": self.page_index,
            "breadcrumb": self.breadcrumb,
            "title": self.title,
            "summary": self.summary,
            "body": self.body,
            "action": [item.to_dict() for item in self.action],
            "action_page": list(self.action_page),
        }


def format_breadcrumb(path: Sequence[int]) -> str:
    """Format a breadcrumb sequence as a path such as /0/1."""

    if not path:
        return "/"
    return "/" + "/".join(str(item) for item in path)
