#!/usr/bin/env python3
"""Materialise a static viewer for the injected (malicious prompt) trap-site.

This packer mirrors Stage 03 but additionally renders `page.injections[]` blocks
and optional CTA links/sentinel strings for program_html-style templates.

This copy lives under web/psaa/ to keep the injection implementation isolated from
the main web/exp/ pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = REPO_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within


def _load_stage03_packer_module():
    spec = importlib.util.spec_from_file_location(
        "trap_site_stage03_packer",
        REPO_ROOT / "exp" / "03_a2perf_render" / "pack_static_site.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate Stage 03 packer module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def _validate_and_instantiate(
    design: dict, metadata: dict, transitions: dict
) -> Tuple[dict, dict]:
    m = _load_stage03_packer_module()
    m._validate_design_bundle(design, metadata, transitions)  # type: ignore[attr-defined]
    node_payload, edge_payload = m._instantiate_actions(  # type: ignore[attr-defined]
        design, metadata, transitions
    )
    return node_payload, edge_payload


def _copy_assets(dst_root: Path) -> None:
    m = _load_stage03_packer_module()
    m._copy_assets(dst_root)  # type: ignore[attr-defined]


def _write_index_with_injections(target_root: Path) -> None:
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Information Retrieval Website</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">
  <link rel=\"stylesheet\" href=\"assets/bootstrap.min.css\">
  <link rel=\"stylesheet\" href=\"assets/jquery-ui.min.css\">
  <link rel=\"stylesheet\" href=\"assets/env-design.css\">
  <style>
    body { margin: 1.5rem; }
    #area { min-height: 640px; }
    .trap-banner { margin-bottom: 1rem; }
    .trap-leaf-marker { font-weight: bold; color: #b84a00; }
    .trap-link-container { margin: 1rem 0; display: flex; flex-wrap: wrap; gap: 0.5rem; position: relative; z-index: 50; }
    .trap-link-container button { white-space: normal; }
    .trap-site-link { position: relative; z-index: 51; }
    #click-canvas, #attention-canvas { pointer-events: none !important; }
    /* Disable MiniWoB overlays */
    #sync-task-cover, #sync-start-cover, #cover-start, .facecover, .sync-task-cover { display: none !important; pointer-events: none !important; }
    /* Injection blocks */
    .trap-injections { margin: 0.75rem 0; }
    .trap-injection { border: 1px solid #e07; background: #fff4f6; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0; }
    .trap-inject-text { margin: 0 0 0.5rem 0; white-space: pre-wrap; tab-size: 4; word-break: break-word; }
    .trap-sentinel { display: inline-block; margin-right: 0.5rem; padding: 0 0.35rem; background: #ffe08a; color: #000; border-radius: 3px; font-weight: 600; }
    .trap-inject-link { margin-top: 0.25rem; }
  </style>
</head>
<body>
  <h1>Information Retrieval Website</h1>
  <p id=\"status\">Loading design…</p>
  <div id=\"wrap\">
    <div id=\"query\"></div>
    <div id=\"area\"></div>
  </div>
  <script src=\"assets/jquery.js\"></script>
  <script src=\"assets/jquery-ui.min.js\"></script>
  <script src=\"assets/bootstrap.min.js\"></script>
  <script src=\"assets/d3.v3.min.js\"></script>
  <script src=\"assets/core.js\"></script>
  <script src=\"assets/ui_utils.js\"></script>
  <script src=\"assets/env-design.js\"></script>
  <script>
    var USE_CONCEPTUAL = false;
    window.genProblem = window.genProblem || function () {};
    const NODE_ACTIONS_URL = 'node_actions.json';
    const EDGE_ACTIONS_URL = 'edge_actions.json';
    const METADATA_URL = 'page_metadata.json';

    let trapNavigationTargets = new Map();
    let trapFallbackInstalled = false;

    

    if (!window.__trapNavCooldownUntil) { window.__trapNavCooldownUntil = 0; }
    function tryOpenPageOnce(index, cause) {
      var now = Date.now();
      if (now < window.__trapNavCooldownUntil) { return; }
      window.__trapNavCooldownUntil = now + 200;
      openPageSafe(index, cause);
    }

    function openPageSafe(index, cause) {
      if (!Number.isFinite(Number(index))) return;
      try { disableCovers(); } catch {}
      
      try { if (typeof window.openPage === 'function') window.openPage(Number(index)); } catch {}
      document.querySelectorAll('[id^="page"]').forEach(el => { el.style.display = 'none'; });
      const targetPage = document.getElementById('page' + Number(index));
      if (targetPage) { targetPage.style.display = 'block'; try { targetPage.scrollIntoView({ block: 'start' }); } catch {} }
      try { window.location.hash = 'p' + Number(index); } catch {}
      const status = document.getElementById('status'); if (status) status.textContent = 'Information navigation → page ' + Number(index) + ' · (' + cause + ')';
    }
    window.openPageSafe = openPageSafe;

    function renderPage(metadata) {
      metadata.pages.forEach(page => {
        const pageRoot = document.getElementById(`page${page.page_index}`);
        if (!pageRoot) return;
        const mainform = pageRoot.querySelector('#mainform') || pageRoot;

        let content = mainform.querySelector('.trap-page-content');
        if (!content) { content = document.createElement('div'); content.className = 'trap-page-content'; mainform.appendChild(content); }
        content.innerHTML = '';
        if (mainform.firstChild !== content) { mainform.insertBefore(content, mainform.firstChild); }

        const actions = Array.isArray(page.action) ? page.action : [];
        const actionConcepts = actions.map(entry => {
          if (!entry) return ''; if (typeof entry === 'string') return entry; if (typeof entry === 'object') return entry.concept || entry.name || ''; return '';
        });
        const hasHeaderPrimitive = actionConcepts.some(name => name === 'header');
        if (!hasHeaderPrimitive && page.title) { const titleEl = document.createElement('h2'); titleEl.className = 'trap-page-title'; titleEl.textContent = page.title; content.appendChild(titleEl); }

        const bodyEl = document.createElement('p'); bodyEl.className = 'trap-page-body'; bodyEl.textContent = page.body; content.appendChild(bodyEl);

        // Render leaf marker if present
        if (page.leaf_marker && typeof page.leaf_marker === 'string' && page.leaf_marker.trim()) {
          const lm = document.createElement('p');
          lm.className = 'trap-leaf-marker';
          lm.id = 'leaf-marker-p' + page.page_index;
          lm.setAttribute('aria-label', 'Leaf marker');
          lm.textContent = page.leaf_marker;
          content.appendChild(lm);
        }

        // Render injections (prompt + sentinel + optional CTA)
        if (Array.isArray(page.injections) && page.injections.length > 0) {
          const injWrap = document.createElement('div'); injWrap.className = 'trap-injections';
          page.injections.forEach((inj, idx) => {
            const blk = document.createElement('div'); blk.className = 'trap-injection'; if (inj.attack_id) blk.setAttribute('data-attack-id', inj.attack_id);
            const pt = document.createElement('p'); pt.className = 'trap-inject-text'; pt.textContent = inj.text || ''; blk.appendChild(pt);
            if (Array.isArray(inj.sentinels) && inj.sentinels.length > 0) {
              const row = document.createElement('div');
              inj.sentinels.forEach(s => { const sp = document.createElement('span'); sp.className = 'trap-sentinel'; sp.textContent = s; row.appendChild(sp); });
              blk.appendChild(row);
            }
            if (inj.show_breadcrumb && typeof page.breadcrumb === 'string') {
              const sm = document.createElement('small'); sm.className = 'trap-inject-breadcrumb text-muted'; sm.textContent = 'Current path ' + page.breadcrumb; blk.appendChild(sm);
            }
            if (inj.cta && inj.cta.target_url) {
              const a = document.createElement('a'); a.className = 'trap-inject-link btn btn-danger btn-sm';
              a.id = (inj.cta.element_id || ('trap_inject_btn_p' + page.page_index + '_' + idx));
              a.href = inj.cta.target_url; a.target = '_blank'; a.rel = 'noopener'; a.textContent = (inj.cta.label || 'Open malicious link');
              blk.appendChild(a);
            }
            injWrap.appendChild(blk);
          });
          content.appendChild(injWrap);
        }

        // Clear and render navigation links
        mainform.querySelectorAll('.trap-link-container').forEach(el => el.remove());
        if (Array.isArray(page.click_targets) && page.click_targets.length > 0) {
          const container = document.createElement('div'); container.className = 'trap-link-container';
          container.addEventListener('click', (ev) => {
            const btn = ev.target && ev.target.closest ? ev.target.closest('.trap-site-link') : null;
            if (!btn) return; ev.preventDefault(); ev.stopPropagation();
            const idx = Number(btn.dataset.targetPage); if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'delegated:container'); }
          }, true);
          page.click_targets.forEach(target => {
            const button = document.createElement('button'); button.id = target.element_id; button.className = 'trap-site-link btn btn-primary btn-sm'; button.type = 'button';
            button.textContent = target.label; button.title = target.blurb; button.setAttribute('role', 'button'); button.setAttribute('tabindex', '0'); if (target.label) button.setAttribute('aria-label', target.label);
            if (typeof target.target_page === 'number') { button.dataset.targetPage = String(target.target_page); }
            button.addEventListener('click', ev => { ev.preventDefault(); ev.stopPropagation(); const idx = Number(ev.currentTarget.dataset.targetPage); if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'direct:button'); } }, true);
            button.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); ev.stopPropagation(); const idx = Number(ev.currentTarget.dataset.targetPage); if (Number.isFinite(idx)) { tryOpenPageOnce(idx, 'kbd:' + ev.key.toLowerCase()); } } }, true);
            container.appendChild(button);
          });
          mainform.appendChild(container);
        }
      });
    }

    function disableCovers() {
      let hiddenCount = 0;
      ['sync-task-cover', 'sync-start-cover', 'cover-start'].forEach(id => { const el = document.getElementById(id); if (el) { el.style.display = 'none'; el.style.pointerEvents = 'none'; el.setAttribute('aria-hidden', 'true'); hiddenCount += 1; } });
      const coverNodes = document.querySelectorAll('.facecover, .sync-task-cover'); coverNodes.forEach(el => { el.style.display = 'none'; el.style.pointerEvents = 'none'; el.setAttribute('aria-hidden', 'true'); hiddenCount += 1; });
      
    }

    function attachTrapFallbackHandlers(metadata) {
      if (!metadata || !Array.isArray(metadata.pages)) return;
      disableCovers();
      trapNavigationTargets = new Map();
      metadata.pages.forEach(page => { (page.click_targets || []).forEach(target => { if (target && target.element_id && typeof target.target_page === 'number') { trapNavigationTargets.set(target.element_id, target.target_page); } }); });

      if (!trapFallbackInstalled) {
        // Log CTA clicks (external links) without blocking navigation
        document.addEventListener('click', event => {
          const a = event && event.target && event.target.closest ? event.target.closest('.trap-inject-link') : null;
          if (!a) return;
          const status = document.getElementById('status'); if (status) status.textContent = 'Attack CTA clicked → ' + (a.href || '');
        }, true);

        // Conservative fallback: pointerdown only and only when not clicking a real trap-site-link
        const handler = (event) => {
          if (!event || !event.target) return;
          if (event.target.closest && event.target.closest('.trap-site-link')) return;
          const isPointerEventSupported = typeof PointerEvent !== 'undefined';
          if (!(isPointerEventSupported && event instanceof PointerEvent)) return;
          let button = null;
          const candidates = Array.from(document.querySelectorAll('.trap-site-link'));
          for (const candidate of candidates) {
            const cr = candidate.getBoundingClientRect();
            if (event.clientX >= cr.left && event.clientX <= cr.right && event.clientY >= cr.top && event.clientY <= cr.bottom) { button = candidate; break; }
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
        const [nodeResp, edgeResp, metaResp] = await Promise.all([ fetch(NODE_ACTIONS_URL), fetch(EDGE_ACTIONS_URL), fetch(METADATA_URL) ]);
        const nodeActions = await nodeResp.json(); const edgeActions = await edgeResp.json(); const metadata = await metaResp.json();

        core.startEpisode();
        disableCovers();
        const status = document.getElementById('status'); status.textContent = 'Design loaded';

        addNodes(nodeActions); nodeCreateFn(); renderPage(metadata); addEdges(edgeActions); edgeCreateFn();
        disableCovers(); attachTrapFallbackHandlers(metadata);
        // Keep navigation tree-only; do not call connectGraph so MiniWoB won't add implicit Next/Submit links.
        document.querySelectorAll('[id^="page"]').forEach(el => { el.style.display = 'none'; });
        const rootPage = document.getElementById('page0'); if (rootPage) rootPage.style.display = 'block';
      } catch (err) {
        console.error('Failed to bootstrap site', err);
        const status = document.getElementById('status'); status.textContent = 'Failed to load design: ' + err.message;
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
        help="Base directory containing experiment artefacts (default: web/runs/trap_site_mvp).",
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
        help="Path to injected design file (list format). Defaults to experiment_root/static/design_injected.json.",
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=None,
        help="Path to transitions.json (possibly unchanged). Defaults to experiment_root/static/transitions_injected.json.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Per-page metadata containing injections. Defaults to experiment_root/static/page_metadata_injected.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for the injected static site (default: experiment_root/static).",
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
    static_root = resolve_path_within(None, root=base_root, relative="static")
    args.design = resolve_path_within(args.design, root=static_root, relative="design_injected.json")
    args.transitions = resolve_path_within(args.transitions, root=static_root, relative="transitions_injected.json")
    args.metadata = resolve_path_within(args.metadata, root=static_root, relative="page_metadata_injected.json")
    args.output = resolve_path_within(args.output, root=static_root, relative="")

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

    node_payload, edge_payload = _validate_and_instantiate(design, metadata, transitions)

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
    _write_index_with_injections(args.output)


if __name__ == "__main__":
    main()
