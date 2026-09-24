"""Drives diagram_recreator over one or more source images end-to-end.

Usage:
    python -m adk_agent.run_pipeline img_7.png [img_8.png ...]
    python -m adk_agent.run_pipeline            # processes every images/*.png

Requires a GOOGLE_API_KEY (from https://aistudio.google.com/apikey) in
adk_agent/diagram_recreator/.env — copy .env.example and fill it in.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
AGENT_DIR = Path(__file__).resolve().parent / "diagram_recreator"


def _load_env():
    env_path = AGENT_DIR / ".env"
    if not env_path.exists():
        return
    import os

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


async def process_image(runner, image_path: Path, output_name: str) -> str:
    from google.adk.sessions import InMemorySessionService  # noqa: F401 (type ref only)
    from google.genai import types

    user_id = "cli"
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id
    )

    image_bytes = image_path.read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    message = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part(
                text=(
                    f"Recreate this diagram as a draw.io file. "
                    f"Use output_name=\"{output_name}\" for save_diagram / render_drawio / validate_drawio / sanity_plot."
                )
            ),
        ],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)
    return final_text


async def main(image_names: list[str]):
    _load_env()
    from google.adk.runners import InMemoryRunner

    from adk_agent.diagram_recreator.agent import root_agent

    runner = InMemoryRunner(agent=root_agent, app_name="diagram_recreator")

    if not image_names:
        image_names = sorted(p.name for p in IMAGES_DIR.glob("*.png"))

    for name in image_names:
        image_path = IMAGES_DIR / name
        if not image_path.exists():
            print(f"skip {name}: not found in {IMAGES_DIR}")
            continue
        output_name = image_path.stem
        print(f"=== {name} -> {output_name}.drawio ===")
        summary = await process_image(runner, image_path, output_name)
        print(summary)
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
