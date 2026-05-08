#!/usr/bin/env python3
"""Materialise a static viewer for the trap-site MVP design."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within

A2PERF_ROOT = WORKSPACE_ROOT / "A2Perf"
ASSET_FILES = {
    "bootstrap/bootstrap.min.css": "assets/bootstrap.min.css",
    "bootstrap/bootstrap.min.js": "assets/bootstrap.min.js",
    "gminiwob/env-design.css": "assets/env-design.css",
    "gminiwob/env-design.js": "assets/env-design.js",
    "miniwob_plusplus/html/core/core.js": "assets/core.js",
    "miniwob_plusplus/html/core/d3.v3.min.js": "assets/d3.v3.min.js",
    "miniwob_plusplus/html/core/jquery-ui/jquery-ui.min.js": "assets/jquery-ui.min.js",
    "miniwob_plusplus/html/core/jquery-ui/jquery-ui.min.css": "assets/jquery-ui.min.css",
    "miniwob_plusplus/html/core/jquery-ui/external/jquery/jquery.js": "assets/jquery.js",
    "miniwob_plusplus/html/common/ui_utils.js": "assets/ui_utils.js",
}

ASSET_DIRS = {
    "gminiwob/images": "assets/images",
}

SUPPRESSED_CONCEPTS = {
    "cart",
    "header_select_items",
    "next",
    "next_checkout",
    "next_login",
    "next_login_page",
    "dealmedia",
    "deck",
    "carousel",
}

_PRIMITIVES_MODULE = None


def _load_stage1_primitives_module():
    """Reuse stage-01 primitive loader so stage outputs stay compatible."""

    global _PRIMITIVES_MODULE
    if _PRIMITIVES_MODULE is not None:
        return _PRIMITIVES_MODULE

    helper_spec = importlib.util.spec_from_file_location(
        "trap_site_stage01_primitives_helper",
        WEB_ROOT / "exp" / "01_node_constructor" / "primitives.py",
    )
    if helper_spec is None or helper_spec.loader is None:
        raise ImportError("Unable to locate stage 01 primitives helper.")

    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules.setdefault(helper_spec.name, helper_module)
    helper_spec.loader.exec_module(helper_module)

    load_fn = getattr(helper_module, "load_primitives_module", None)
    if load_fn is None:
        raise AttributeError("Stage 01 primitives helper missing load_primitives_module().")

    _PRIMITIVES_MODULE = load_fn()
    return _PRIMITIVES_MODULE


def _validate_design_bundle(design: dict, metadata: dict, transitions: dict | None) -> None:
    """Ensure stage-02 artefacts remain internally consistent before packing."""

    num_pages = design.get("number_of_pages")
    if not isinstance(num_pages, int) or num_pages < 1:
        raise ValueError("Design number_of_pages must be a positive integer.")

    pages = metadata.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Metadata must contain a 'pages' array.")
    if len(pages) != num_pages:
        raise ValueError(
            f"Metadata provides {len(pages)} pages but design declares {num_pages}."
        )

    seen_indices = set()
    click_targets: Dict[str, Tuple[int, str]] = {}
    for expected_index in range(num_pages):
        try:
            page = pages[expected_index]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Metadata pages missing entry for index {expected_index}.") from exc

        page_index = page.get("page_index")
        if page_index != expected_index:
            raise ValueError(
                f"Metadata pages must be sorted by page_index; expected {expected_index}, got {page_index}."
            )
        seen_indices.add(page_index)

        for target in page.get("click_targets") or []:
            element_id = target.get("element_id")
            target_page = target.get("target_page")
            if not element_id:
                raise ValueError(f"Page {page_index} defines a click_target without element_id.")
            if element_id in click_targets:
                owner_page, _ = click_targets[element_id]
                raise ValueError(
                    f"Duplicate click_target element_id '{element_id}' found on pages {owner_page} and {page_index}."
                )
            if not isinstance(target_page, int) or not (0 <= target_page < num_pages):
                raise ValueError(
                    f"Click target '{element_id}' on page {page_index} references invalid target_page {target_page}."
                )
            click_targets[element_id] = (page_index, page.get("breadcrumb", ""))

    if seen_indices != set(range(num_pages)):
        raise ValueError("Metadata does not cover a contiguous range of page_index values.")

    if not transitions:
        return

    for action in transitions.get("actions", []):
        if not isinstance(action, dict):
            raise ValueError("Transitions must be dictionaries.")
        target_page = action.get("target_page")
        if target_page is not None and (not isinstance(target_page, int) or target_page >= num_pages):
            raise ValueError(
                f"Transition {action} references invalid target_page {target_page} (num_pages={num_pages})."
            )
        source_group = action.get("source_group")
        if (
            isinstance(source_group, str)
            and not source_group.startswith("group")
            and not source_group.startswith("actionable_")
        ):
            if source_group not in click_targets:
                raise ValueError(
                    f"Transition requires source_group '{source_group}' not present in metadata click_targets."
                )


def _normalize_design_action(action_entry) -> tuple[str, Dict[str, object]]:
    if isinstance(action_entry, str):
        return action_entry.strip(), {}
    if isinstance(action_entry, dict):
        concept = str(action_entry.get("concept", action_entry.get("name", ""))).strip()
        if not concept:
            raise ValueError("Design action entry missing concept field.")
        props_obj = action_entry.get("props")
        if props_obj is None:
            props = {str(k): v for k, v in action_entry.items() if k not in {"concept"}}
        elif isinstance(props_obj, dict):
            props = {str(k): v for k, v in props_obj.items()}
        else:
            raise ValueError("Design action props must be an object.")
        return concept, props
    concept = str(action_entry).strip()
    return concept, {}


def _merge_controls(entry_json: dict, props: Dict[str, object]) -> None:
    for key, value in props.items():
        if key == "controls" and isinstance(value, dict):
            controls = entry_json.setdefault("controls", {})
            if not isinstance(controls, dict):
                entry_json["controls"] = {}
                controls = entry_json["controls"]
            controls.update(value)
        else:
            entry_json[key] = value


def _instantiate_actions(design: dict, metadata: dict, transitions: dict | None) -> Tuple[dict, dict]:
    web_primitives = _load_stage1_primitives_module()
    primitives = web_primitives.CONCEPTS2DESIGN
    page_placeholder = web_primitives.PAGE_PH

    pages = metadata.get("pages", [])
    page_lookup = {page.get("page_index"): page for page in pages if isinstance(page, dict)}

    node_actions = []
    edge_actions = []
    for action_entry, page in zip(design["action"], design["action_page"]):
        concept, props = _normalize_design_action(action_entry)
        if not concept:
            continue
        if concept in SUPPRESSED_CONCEPTS:
            continue
        try:
            template = primitives[concept]
        except KeyError as exc:
            raise KeyError(f"Unknown concept '{concept}' referenced in design.") from exc
        if isinstance(template, list):
            entries = template
        else:
            entries = (template,)
        for raw_entry in entries:
            entry_json = json.loads(raw_entry.replace(page_placeholder, str(page)))
            _merge_controls(entry_json, props)

            page_meta = page_lookup.get(page, {})
            controls = entry_json.setdefault("controls", {})
            if concept == "header":
                controls["headerText"] = page_meta.get("title", controls.get("headerText", ""))
            elif concept == "navbar":
                candidate_items = [
                    target.get("label")
                    for target in page_meta.get("click_targets") or []
                    if isinstance(target, dict) and target.get("label")
                ]
                if candidate_items:
                    controls["menuItems"] = candidate_items
                title = page_meta.get("title")
                if isinstance(title, str) and title.strip():
                    controls["homeLink"] = title.strip()

            if entry_json.get("is_transition"):
                edge_actions.append(entry_json)
            else:
                node_actions.append(entry_json)

    if transitions:
        edge_actions.extend(transitions.get("actions", []))

    node_payload = {
        "num_pages": design["number_of_pages"],
        "actions": node_actions,
    }
    edge_payload = {"actions": edge_actions}
    return node_payload, edge_payload


def _copy_assets(target_root: Path) -> None:
    src_base = (
        A2PERF_ROOT / "a2perf" / "domains" / "web_navigation" / "gwob"
    )
    for relative, destination in ASSET_FILES.items():
        src = src_base / relative
        dst = target_root / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for relative, destination in ASSET_DIRS.items():
        src = src_base / relative
        dst = target_root / destination
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _write_index(target_root: Path) -> None:
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Information Retrieval Website</title>
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <link rel="stylesheet" href="assets/bootstrap.min.css">
  <link rel="stylesheet" href="assets/jquery-ui.min.css">
  <link rel="stylesheet" href="assets/env-design.css">
  <style>
    body { margin: 1.5rem; }
    #area { min-height: 640px; }
    .trap-banner { margin-bottom: 1rem; }
    .trap-leaf-marker { font-weight: bold; color: #b84a00; }
    .trap-link-container {
      margin: 1rem 0;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      position: relative;
      z-index: 50;
    }
    .trap-link-container button {
      white-space: normal;
    }
    .trap-site-link {
      position: relative;
      z-index: 51;
    }
    #click-canvas,
    #attention-canvas {
      pointer-events: none !important;
    }
    /* Disable MiniWoB start/sync overlays so automation clicks reach buttons */
    #sync-task-cover, #sync-start-cover, #cover-start, .facecover, .sync-task-cover {
      display: none !important;
      pointer-events: none !important;
    }
  </style>
</head>
<body>
  <h1>Information Retrieval Website</h1>
  <p id="status">Loading design…</p>
  <div id="wrap">
    <div id="query"></div>
    <div id="area"></div>
  </div>
  <script src="assets/jquery.js"></script>
  <script src="assets/jquery-ui.min.js"></script>
  <script src="assets/bootstrap.min.js"></script>
  <script src="assets/d3.v3.min.js"></script>
  <script src="assets/core.js"></script>
  <script src="assets/ui_utils.js"></script>
  <script src="assets/env-design.js"></script>
  <script>
    // Ensure env-design.js optional conceptual flag exists to avoid ReferenceError
    // Some builds of env-design.js check a global USE_CONCEPTUAL variable.
    // Default to false for our static trap-site.
    var USE_CONCEPTUAL = false;
    // MiniWob core expects a global genProblem() to exist when clicking START.
    // Our static viewer doesn't use it, so provide a safe no-op to allow
    // the START overlay to dismiss on click.
    window.genProblem = window.genProblem || function () {};
    const NODE_ACTIONS_URL = 'node_actions.json';
    const EDGE_ACTIONS_URL = 'edge_actions.json';
    const METADATA_URL = 'page_metadata.json';

    let trapNavigationTargets = new Map();
    let trapFallbackInstalled = false;

    

    // Simple global cooldown to avoid multiple navigations per user gesture
    if (!window.__trapNavCooldownUntil) { window.__trapNavCooldownUntil = 0; }
    function tryOpenPageOnce(index, cause) {
      var now = Date.now();
      if (now < window.__trapNavCooldownUntil) { return; }
      window.__trapNavCooldownUntil = now + 200; // 200ms guard per gesture
      openPageSafe(index, cause);
    }

    function openPageSafe(index, cause) {
      if (!Number.isFinite(Number(index))) {
        return;
      }
      try {
        disableCovers();
      } catch (err) {
        console.warn('disableCovers failed', err);
      }
      
      try {
        if (typeof window.openPage === 'function') {
          window.openPage(Number(index));
        }
      } catch (err) {
        console.error('openPage invocation failed:', err);
      }
      document.querySelectorAll('[id^="page"]').forEach(el => {
        el.style.display = 'none';
      });
      const targetPage = document.getElementById('page' + Number(index));
      if (targetPage) {
        targetPage.style.display = 'block';
        if (typeof targetPage.scrollIntoView === 'function') {
          try {
            targetPage.scrollIntoView({ block: 'start' });
          } catch (err) {
            console.warn('scrollIntoView failed', err);
          }
        }
      }
      try {
        window.location.hash = 'p' + Number(index);
      } catch (err) {
        console.warn('hash update failed', err);
      }
      const status = document.getElementById('status');
      if (status) {
        status.textContent = 'Information navigation → page ' + Number(index) + ' · (' + cause + ')';
      }
    }
    window.openPageSafe = openPageSafe;

    function renderPage(metadata) {
      metadata.pages.forEach(page => {
        const pageRoot = document.getElementById(`page${page.page_index}`);
        if (!pageRoot) return;
        const mainform = pageRoot.querySelector('#mainform') || pageRoot;

        let content = mainform.querySelector('.trap-page-content');
        if (!content) {
          content = document.createElement('div');
          content.className = 'trap-page-content';
          mainform.appendChild(content);
        }
        content.innerHTML = '';
        if (mainform.firstChild !== content) {
          mainform.insertBefore(content, mainform.firstChild);
        }

        const actions = Array.isArray(page.action) ? page.action : [];
        const actionConcepts = actions.map(entry => {
          if (!entry) return '';
          if (typeof entry === 'string') return entry;
          if (typeof entry === 'object') return entry.concept || entry.name || '';
          return '';
        });
        const hasHeaderPrimitive = actionConcepts.some(name => name === 'header');

        if (!hasHeaderPrimitive && page.title) {
          const titleEl = document.createElement('h2');
          titleEl.className = 'trap-page-title';
          titleEl.textContent = page.title;
          content.appendChild(titleEl);
        }

        const bodyEl = document.createElement('p');
        bodyEl.className = 'trap-page-body';
        bodyEl.textContent = page.body;
        content.appendChild(bodyEl);

        // Render leaf marker if present
        if (page.leaf_marker && typeof page.leaf_marker === 'string' && page.leaf_marker.trim()) {
          const lm = document.createElement('p');
          lm.className = 'trap-leaf-marker';
          lm.id = 'leaf-marker-p' + page.page_index;
          lm.setAttribute('aria-label', 'Leaf marker');
          lm.textContent = page.leaf_marker;
          content.appendChild(lm);
        }

        mainform.querySelectorAll('.trap-link-container').forEach(el => el.remove());

        if (Array.isArray(page.click_targets) && page.click_targets.length > 0) {
          const container = document.createElement('div');
          container.className = 'trap-link-container';
          container.addEventListener('click', (ev) => {
            const btn = ev.target && ev.target.closest ? ev.target.closest('.trap-site-link') : null;
            if (!btn) return;
            ev.preventDefault();
            ev.stopPropagation();
            const idx = Number(btn.dataset.targetPage);
            if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'delegated:container'); }
          }, true);
          page.click_targets.forEach(target => {
            const button = document.createElement('button');
            button.id = target.element_id;
            button.className = 'trap-site-link btn btn-primary btn-sm';
            button.type = 'button';
            button.textContent = target.label;
            button.title = target.blurb;
            button.setAttribute('role', 'button');
            button.setAttribute('tabindex', '0');
            if (target.label) {
              button.setAttribute('aria-label', target.label);
            }
            if (typeof target.target_page === 'number') {
              button.dataset.targetPage = String(target.target_page);
            }
            button.addEventListener('click', ev => {
              ev.preventDefault();
              ev.stopPropagation();
              const idx = Number(ev.currentTarget.dataset.targetPage);
            if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'direct:button'); }
          }, true);
            button.addEventListener('keydown', ev => {
              if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                ev.stopPropagation();
                const idx = Number(ev.currentTarget.dataset.targetPage);
                if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'kbd:' + ev.key.toLowerCase()); }
              }
            }, true);
            container.appendChild(button);
          });
          mainform.appendChild(container);
        }
      });
    }

    function disableCovers() {
      let hiddenCount = 0;
      ['sync-task-cover', 'sync-start-cover', 'cover-start'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.style.display = 'none';
          el.style.pointerEvents = 'none';
          el.setAttribute('aria-hidden', 'true');
          hiddenCount += 1;
        }
      });
      const coverNodes = document.querySelectorAll('.facecover, .sync-task-cover');
      coverNodes.forEach(el => {
        el.style.display = 'none';
        el.style.pointerEvents = 'none';
        el.setAttribute('aria-hidden', 'true');
        hiddenCount += 1;
      });
      
    }

    function attachTrapFallbackHandlers(metadata) {
      if (!metadata || !Array.isArray(metadata.pages)) {
        return;
      }
      disableCovers();
      trapNavigationTargets = new Map();
      metadata.pages.forEach(page => {
        (page.click_targets || []).forEach(target => {
          if (target && target.element_id && typeof target.target_page === 'number') {
            trapNavigationTargets.set(target.element_id, target.target_page);
          }
        });
      });
      if (trapNavigationTargets.size === 0) { return; }

      if (!trapFallbackInstalled) {
        // Only one conservative fallback: pointerdown anywhere near a button,
        // and only if the original target is NOT already a trap-site-link
        const handler = (event) => {
          if (!event || !event.target) return;
          // If event already targets a real trap-site-link, let its own handler run
          if (event.target.closest && event.target.closest('.trap-site-link')) return;
          const isPointerEventSupported = typeof PointerEvent !== 'undefined';
          if (!(isPointerEventSupported && event instanceof PointerEvent)) return;
          let button = null;
          const candidates = Array.from(document.querySelectorAll('.trap-site-link'));
          for (const candidate of candidates) {
            const cr = candidate.getBoundingClientRect();
            if (
              event.clientX >= cr.left && event.clientX <= cr.right &&
              event.clientY >= cr.top && event.clientY <= cr.bottom
            ) { button = candidate; break; }
          }
          if (!button) return;
          const mapped = trapNavigationTargets.get(button.id);
          const datasetIdx = Number(button.dataset.targetPage);
          const targetPageIndex = typeof mapped === 'number' ? mapped : datasetIdx;
          if (!Number.isFinite(targetPageIndex)) return;
          event.stopPropagation();
          tryOpenPageOnce(targetPageIndex, 'fallback:pointerdown');
        };
        document.addEventListener('pointerdown', handler, true);
        trapFallbackInstalled = true;
      }
    }

    async function bootstrapSite() {
      try {
        const [nodeResp, edgeResp, metaResp] = await Promise.all([
          fetch(NODE_ACTIONS_URL),
          fetch(EDGE_ACTIONS_URL),
          fetch(METADATA_URL),
        ]);
        const nodeActions = await nodeResp.json();
        const edgeActions = await edgeResp.json();
        const metadata = await metaResp.json();

        core.startEpisode();
        disableCovers();
        const status = document.getElementById('status');
        status.textContent = 'Design loaded';

        addNodes(nodeActions);
        nodeCreateFn();
        renderPage(metadata);
        addEdges(edgeActions);
        edgeCreateFn();
        disableCovers();
        attachTrapFallbackHandlers(metadata);
        // Keep navigation tree-only; do not call connectGraph so MiniWoB won't add implicit Next/Submit links.
        document.querySelectorAll('[id^="page"]').forEach(el => { el.style.display = 'none'; });
        const rootPage = document.getElementById('page0');
        if (rootPage) {
          rootPage.style.display = 'block';
        }
      } catch (err) {
        console.error('Failed to bootstrap site', err);
        const status = document.getElementById('status');
        status.textContent = 'Failed to load design: ' + err.message;
      }
    }

    window.addEventListener('DOMContentLoaded', bootstrapSite);
  </script>
</body>
</html>
"""
    (target_root / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Base directory containing Stage 02 outputs (default: web/runs/trap_site_mvp).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional experiment identifier appended under --experiment-root.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=None,
        help="Path to website_designs.json (list format). Defaults to experiment_root/website_designs.json.",
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=None,
        help="Path to transitions.json. Defaults to experiment_root/transitions.json.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Per-page metadata description. Defaults to experiment_root/page_metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for the static site (default: experiment_root/static).",
    )
    parser.add_argument(
        "--design-index",
        type=int,
        default=0,
        help="Which design entry to materialise (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.design = resolve_path_within(args.design, root=base_root, relative="website_designs.json")
    args.transitions = resolve_path_within(args.transitions, root=base_root, relative="transitions.json")
    args.metadata = resolve_path_within(args.metadata, root=base_root, relative="page_metadata.json")
    args.output = resolve_path_within(args.output, root=base_root, relative="static")

    args.output.mkdir(parents=True, exist_ok=True)

    designs = json.loads(args.design.read_text(encoding="utf-8"))
    if not designs:
        raise ValueError("Design file is empty.")
    if args.design_index >= len(designs):
        raise IndexError(f"Design index {args.design_index} out of range (len={len(designs)}).")
    design = designs[args.design_index]

    if args.transitions.exists():
        transitions = json.loads(args.transitions.read_text(encoding="utf-8"))
    else:
        transitions = {"actions": []}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    _validate_design_bundle(design, metadata, transitions)
    node_payload, edge_payload = _instantiate_actions(design, metadata, transitions)

    (args.output / "design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "transitions.json").write_text(
        json.dumps(transitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "page_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "node_actions.json").write_text(
        json.dumps(node_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "edge_actions.json").write_text(
        json.dumps(edge_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _copy_assets(args.output)
    _write_index(args.output)


if __name__ == "__main__":
    main()
