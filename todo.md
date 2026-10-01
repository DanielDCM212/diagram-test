# TODO: diagram_editor agent

Status of the agent: built and unit-tested (111 tests), **not yet run with the real model**.
Priority order inside each section. Tick items as they are done.

## Next up: make it safe to try on real diagrams
- [ ] **Real runs.** Put `ANTHROPIC_API_KEY` in `adk_agent/diagram_editor/.env`, run one edit and one create
      (`python -m adk_agent.run_editor ...`), tune the instructions from what the model gets wrong.
- [ ] **Test set for the agent.** 10-15 request + diagram pairs (rename, add a cache, fix overlaps, AWS background...)
      with simple pass/fail checks (untouched cells unchanged, result validates). Re-run after every prompt/tool change.
- [ ] **Undo and versions.** Keep each diagram state in the session; add `undo` and "what changed". Fits the existing
      `iag_diagram_versions` table later.
- [x] **Deterministic change summary.** Computed by code, not the model: "renamed 1, added 2, moved 4, removed 0".
      Done: `diff.py`, `get_change_summary` tool, per-call `changes` in tool results, `state["change_summary"]`,
      printed by `run_editor`. Still to verify in a real run: that the model quotes it verbatim instead of paraphrasing.

## Inputs we are waiting on
- [ ] **Reference diagrams** into `adk_agent/diagram_editor/reference_diagrams/` (varied: simple, dense, nested zones).
      Then check what `derive_profile` makes of the real ones (zone detection was only tried on synthetic files).
- [ ] **Export server check.** Set `DRAWIO_EXPORT_URL`, confirm `renderer: export_server` in the preview result.
      (Only tested against a fake server so far; the real one uses `xml`/`format`/`from` form fields.)
- [ ] Confirm the preview image actually reaches the model in a live run (same mechanism as diagram_recreator).
- [ ] Confirm `adk web adk_agent` lists `diagram_editor`, and what mime type a `.drawio` attachment gets from the browser.

## Better results
- [ ] **Before/after preview** with changed elements highlighted.
- [ ] **House-style check:** compare any diagram to the reference conventions, report deviations, optional "make it conform".
- [ ] **Clarifying questions** when a request matches several candidates ("the AWS zone" with several groups): a tool that
      returns candidates + an instruction to ask.

## More capabilities
- [ ] Zone/group operations in edit mode (create a house-styled zone, re-parent a set of nodes into it).
- [ ] Alignment and spacing: align, distribute evenly, snap to grid, consistent sizes.
- [ ] Arrow routing: detect arrows crossing boxes, re-route; optionally attach loose arrow ends to the nearest node.
- [ ] Bulk selection by label pattern / shape / color instead of listing ids.
- [ ] Edit operations on pages other than the first (describe/preview already take a page index).
- [ ] Resolve `%placeholder%` labels (shown raw today).

## Overlap clean-up: known limits (see `docs/overlap_resolution_limits.drawio`)
- [ ] Clearance between boxes is only added when a pair is pushed; chains of pushes can leave neighbours closer than
      the usual spacing (page 9). Consider a final clearance pass.
- [ ] Round/diamond/rotated shapes are measured by bounding box (page 8). Consider a shape-aware test.
- [ ] Arrows through boxes and loose arrow ends are not handled (pages 6-7), see arrow routing above.
- [ ] Regenerate the limits document after any change to the tool: `python -m scripts.build_overlap_limits_doc`
      (it fails if the tool stops behaving as the pages claim).

## For the parent agent (later)
- [ ] Typed result `{xml, summary, warnings}` instead of text + `state["drawio_xml"]` only.
- [ ] Attach as a sub-agent (`description` is the routing text; result key is `drawio_xml`).
- [ ] Optional persistence step reusing `persist_diagram`.

## Housekeeping
- [ ] Nothing from this work is committed yet (`adk_agent/diagram_editor/`, `adk_agent/run_editor.py`, `tests/`, `docs/`,
      `scripts/build_overlap_limits_doc.py`, changes to `pyproject.toml` and `.gitignore`).
- [ ] Test fakes must use ADK's real `State` (no `pop`/`del`/`items`): already done in `tests/diagram_editor/test_tools.py`.
