"""Drives diagram_editor from the command line.

Usage:
    python -m adk_agent.run_editor "create a diagram of an order system with web, API, Postgres"
    python -m adk_agent.run_editor --xml diagrams/img_3.drawio "rename the API box to Gateway"
    python -m adk_agent.run_editor --xml a.drawio --name out "add a Redis cache next to the API"

Writes the resulting XML (from session state `drawio_xml`) to diagramsEditor/<name>.drawio.
Requires ANTHROPIC_API_KEY in adk_agent/diagram_editor/.env (copy .env.example), or in the
environment. House-style references are read from adk_agent/diagram_editor/reference_diagrams/
(override with DIAGRAM_REFERENCE_DIR).
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(__file__).resolve().parent / "diagram_editor"
OUT_DIR = PROJECT_ROOT / "diagramsEditor"


def _load_env():
    env_path = AGENT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


async def run(request: str, xml_path: Path | None, name: str) -> str:
    from google.genai import types
    from google.adk.runners import InMemoryRunner

    from adk_agent.diagram_editor.agent import root_agent

    runner = InMemoryRunner(agent=root_agent, app_name="diagram_editor")
    user_id = "cli"
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id=user_id)

    parts = []
    if xml_path:
        parts.append(types.Part.from_bytes(data=xml_path.read_bytes(), mime_type="application/xml"))
    parts.append(types.Part(text=request))

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=types.Content(role="user", parts=parts)
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    result = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session.id
    )
    xml = result.state.get("drawio_xml") if result else None
    changes = result.state.get("change_summary") if result else None
    if changes:
        # computed from the real before/after diagram, independent of whatever the model wrote above
        final_text += "\n\n--- Verified changes (computed by code) ---\n" + changes["summary"]
        final_text += "".join(f"\n  - {ln}" for ln in changes.get("lines", []))
    if xml:
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"{name}.drawio"
        out.write_text(xml, encoding="utf-8")
        final_text += f"\n\n[wrote {out}]"
    else:
        final_text += "\n\n[no diagram was produced]"
    return final_text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", help="what to do, in plain language")
    parser.add_argument("--xml", type=Path, help="existing .drawio file to edit (omit to create a new diagram)")
    parser.add_argument("--name", help="output file name without extension (default: input stem, or 'new_diagram')")
    args = parser.parse_args()

    if args.xml and not args.xml.is_file():
        sys.exit(f"not found: {args.xml}")
    _load_env()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = args.name or (f"{args.xml.stem}_edited" if args.xml else "new_diagram")
    print(asyncio.run(run(args.request, args.xml, name)))


if __name__ == "__main__":
    main()
