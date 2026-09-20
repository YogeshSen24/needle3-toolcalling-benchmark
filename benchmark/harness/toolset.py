"""Tool catalogue assembly: size, distractor similarity, gold position, prefilter.

Gold-tool position is a controlled factor, not an incidental detail. During
design, the same query at a 50-tool catalogue succeeded when the gold tool sat
last and failed at positions 0, 10 and 25. Leaving ordering to chance would
have injected that variance into every other measurement.
"""
from __future__ import annotations

import random

from .embedding import CACHE, serialize_tool

_LOW = [
    ("export_report", "Export a report to a file by its identifier."),
    ("compile_firmware", "Compile a firmware image for a target board."),
    ("translate_contract", "Translate a contract document into another language."),
    ("transcribe_podcast", "Transcribe a podcast episode to text."),
    ("validate_dataset", "Validate a dataset against its schema."),
    ("publish_article", "Publish a drafted article."),
    ("encrypt_backup", "Encrypt a backup archive."),
    ("resize_image", "Resize an image to given dimensions."),
    ("audit_ledger", "Audit a financial ledger for a period."),
    ("index_corpus", "Build a search index over a document corpus."),
]

_MEDIUM = [
    ("lookup_account", "Look up an account by its identifier."),
    ("lookup_subscription", "Look up a subscription by its identifier."),
    ("lookup_ticket", "Look up a support ticket by its identifier."),
    ("lookup_shipment", "Look up a shipment by its identifier."),
    ("lookup_warranty", "Look up a warranty by its identifier."),
    ("lookup_licence", "Look up a licence by its identifier."),
    ("lookup_payment", "Look up a payment by its identifier."),
    ("lookup_refund", "Look up a refund by its identifier."),
    ("lookup_quote", "Look up a quote by its identifier."),
    ("lookup_lead", "Look up a sales lead by its identifier."),
]

_HIGH = [
    ("set_room_scene", "Apply a saved lighting scene to a room."),
    ("set_room_mode", "Set the operating mode of a room."),
    ("set_room_preset", "Apply a saved preset to a room."),
    ("set_room_schedule", "Set the daily schedule for a room."),
    ("set_room_zone", "Assign a room to a control zone."),
    ("set_device_level", "Set the output level of a device in a room."),
    ("set_device_state", "Set the on or off state of a device in a room."),
    ("set_device_timer", "Set a countdown timer for a device in a room."),
    ("set_ambient_target", "Set the ambient comfort target for a room."),
    ("set_comfort_profile", "Apply a comfort profile to a room."),
]

BANKS = {"low": _LOW, "medium": _MEDIUM, "high": _HIGH}


def _filler(name: str, description: str) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object",
                           "properties": {"item_id": {"type": "string",
                                                      "description": "The identifier."}},
                           "required": ["item_id"]}}


def build_catalogue(env, profile, *, size: int | None = None, similarity: str = "low",
                    gold_position: int | None = None, gold_names: list[str] | None = None,
                    seed: int = 0, prompt: str | None = None) -> list[dict]:
    core = [dict(t) for t in env.tools(profile)]

    if size is not None and size > len(core):
        bank = BANKS[similarity]
        fillers, i = [], 0
        while len(fillers) < size - len(core):
            name, desc = bank[i % len(bank)]
            suffix = "" if i < len(bank) else f"_{i // len(bank) + 1}"
            fillers.append(_filler(name + suffix, desc))
            i += 1
        tools = core + fillers
    elif size is not None and size < len(core):
        # Never drop a gold tool to hit a target size: that would measure the
        # harness removing the answer, not the model failing to find it.
        keep = set(gold_names or [])
        others = [t for t in core if t["name"] not in keep]
        tools = [t for t in core if t["name"] in keep] + others[:max(0, size - len(keep))]
    else:
        tools = core

    rng = random.Random(seed)
    rng.shuffle(tools)

    # Explicit prefilter (the opt_prefilter profile): keep the top-k by
    # embedding similarity to the request, always retaining nothing by name —
    # the filter must be able to drop the gold tool, or it is not a real test.
    if profile.prefilter and prompt and len(tools) > profile.prefilter:
        by_text = {serialize_tool(t): t for t in tools}
        ranked = CACHE.rank(prompt, list(by_text))
        if ranked and any(score for score, _ in ranked):
            tools = [by_text[text] for _, text in ranked[:profile.prefilter]]

    if gold_position is not None and gold_names:
        gold_set = set(gold_names)
        gold = [t for t in tools if t["name"] in gold_set]
        rest = [t for t in tools if t["name"] not in gold_set]
        pos = max(0, min(gold_position, len(rest)))
        tools = rest[:pos] + gold + rest[pos:]
    return tools


def catalogue_texts(env, profile) -> list[str]:
    """Serialized schemas, for pre-warming the embedding cache."""
    return [serialize_tool(t) for t in env.tools(profile)]
