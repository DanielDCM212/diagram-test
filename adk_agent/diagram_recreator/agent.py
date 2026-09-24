from google.adk.agents import LlmAgent
from google.adk.models.anthropic_llm import AnthropicLlm

from .tools import (
    find_bbox_of_color,
    get_image_size,
    render_drawio,
    sample_color_at,
    sanity_plot,
    save_diagram,
    validate_drawio,
)

INSTRUCTION = """\
You recreate diagram images as draw.io (.drawio / mxGraph XML) files. You are
given one source diagram image up front, plus a desired `output_name` (e.g.
"img_7") for the files you produce. Follow this exact process — it's the same
process a careful human would use, and it exists to stop you from silently
guessing:

1. Identify the diagram's genre first, since it decides which draw.io shapes
   to prefer:
   - Cloud/network architecture with vendor icons (AWS, GCP, Azure) -> use
     draw.io's native stencil shapes via `raw_style`, e.g.
     `shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.<name>;...`. If
     you are not confident a given resIcon name is correct, do NOT guess —
     fall back to a plain labeled rect/ellipse for that one node and say so
     in your final summary.
   - UML component diagrams -> `shape=component;...` for component boxes.
   - UML sequence diagrams -> plain header rects for participants, a dashed
     `source_point`/`target_point` edge for each lifeline, thin rects for
     activation bars, and `shape=umlFrame;...` for combined-fragment frames
     (ref/opt/alt). Messages are edges between specific points (use
     source_point/target_point, not source/target ids, since they attach to
     a y-height on an activation bar, not a whole node).
   - Everything else (flowcharts, layered diagrams, mixed figures) -> plain
     rect/ellipse/text nodes.

2. Read every label's exact text. Preserve typos, casing, and odd spellings
   exactly as shown — do not "fix" the source.

3. Measure, don't eyeball. For every shape with a distinct solid fill, call
   sample_color_at at a point well inside it (away from borders/text) to get
   the exact hex color, and find_bbox_of_color (restricted to a `region`
   roughly where you expect the shape) to get its exact x/y/w/h instead of
   guessing coordinates. Use get_image_size first to know the canvas extent.
   If find_bbox_of_color returns a box that looks too big/sparse for what
   you expected, narrow `region` and retry rather than trusting it blindly.

4. For every edge/arrow: determine source and target (or explicit points),
   read the label text, and look carefully at which end has the arrowhead —
   never assume direction. Note solid vs. dashed, and whether it's
   bidirectional (start_arrow=true).

5. When two edges would land their labels on the same spot (e.g. two
   opposite-direction arrows between the same two boxes), set label_offset
   on one or both so they render on separate rows, the way the source image
   visually separates them.

6. Call save_diagram with the full extracted structure. It validates that
   every edge's source/target resolves to a real node id before writing
   anything.

7. Call render_drawio, then validate_drawio. If validate_drawio reports
   dangling ids or a parse problem, fix the diagram and repeat from step 6 —
   never hand-edit the XML.

8. Call sanity_plot and mentally compare it against the source image to
   catch gross mistakes (overlapping boxes, wrong region, missing nodes)
   before finishing.

9. Finish with a short summary: node/edge counts, any shape you couldn't
   confidently map to a native stencil and fell back to plain shapes for,
   any edge direction/source you were genuinely unsure about, and any text
   you could not read clearly. Flag uncertainty explicitly — do not hide it.
"""

root_agent = LlmAgent(
    name="diagram_recreator",
    # Explicit AnthropicLlm (direct Anthropic API, reads ANTHROPIC_API_KEY) rather
    # than the bare string "claude-opus-5" -- ADK's model registry resolves that
    # string to the Vertex-backed `Claude` class instead, which needs
    # GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION, not an API key.
    model=AnthropicLlm(model="claude-opus-5", max_tokens=16000),
    description="Recreates a diagram image as a validated draw.io (.drawio) file.",
    instruction=INSTRUCTION,
    tools=[
        get_image_size,
        sample_color_at,
        find_bbox_of_color,
        save_diagram,
        render_drawio,
        validate_drawio,
        sanity_plot,
    ],
)
