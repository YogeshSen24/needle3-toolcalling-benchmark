"""Tier 2 — workspace (calendar + files).

Harder than tier 1 along properties that can be counted rather than asserted:
twelve tools instead of ten, two entity types that share verbs (`find_events`
vs `find_file`, `move_event` vs `move_file`), opaque identifiers that must be
carried from a lookup into a mutation, and dates and durations that have a
required format. The verb collisions are the point: this is where a router has
to use the object, not just the verb.
"""
from __future__ import annotations

from ._schema import apply_overrides, param, tool

ENV_ID = "workspace"
TIER = 2

FOLDERS = ["finance", "product", "team", "archive"]
READ_ONLY = {"find_events", "get_event", "find_free_slot", "find_file", "read_file"}

TODAY = "2026-09-20"
TOMORROW = "2026-09-21"


def initial_state(overrides: dict | None = None) -> dict:
    return apply_overrides({
        "events": {
            "ev_101": {"title": "Design review", "date": TOMORROW, "start": "10:00",
                       "duration": 60, "cancelled": False},
            "ev_102": {"title": "1:1 with Priya", "date": TOMORROW, "start": "14:00",
                       "duration": 30, "cancelled": False},
            "ev_103": {"title": "Sprint planning", "date": "2026-09-22", "start": "09:00",
                       "duration": 90, "cancelled": False},
        },
        "files": {
            "f_201": {"name": "q3_budget.xlsx", "folder": "finance", "deleted": False,
                      "shared_with": []},
            "f_202": {"name": "launch_plan.docx", "folder": "product", "deleted": False,
                      "shared_with": []},
            "f_203": {"name": "retro_notes.md", "folder": "team", "deleted": False,
                      "shared_with": []},
        },
    }, overrides)


# --------------------------------------------------------------------------
# Executors
# --------------------------------------------------------------------------

def _find_events(s, date):
    hits = [{"event_id": k, **v} for k, v in s["events"].items()
            if v["date"] == date and not v["cancelled"]]
    return {"date": date, "count": len(hits), "events": hits}


def _get_event(s, event_id):
    ev = s["events"].get(event_id)
    return {"event_id": event_id, **ev} if ev else {"error": f"no such event: {event_id}"}


def _create_event(s, title, date, start_time, duration_minutes):
    new_id = f"ev_{100 + len(s['events']) + 1}"
    s["events"][new_id] = {"title": title, "date": date, "start": start_time,
                           "duration": int(duration_minutes), "cancelled": False}
    return {"ok": True, "event_id": new_id}


def _move_event(s, event_id, new_date, new_time):
    ev = s["events"].get(event_id)
    if not ev:
        return {"error": f"no such event: {event_id}"}
    ev["date"], ev["start"] = new_date, new_time
    return {"ok": True, "event_id": event_id, **ev}


def _cancel_event(s, event_id):
    ev = s["events"].get(event_id)
    if not ev:
        return {"error": f"no such event: {event_id}"}
    ev["cancelled"] = True
    return {"ok": True, "event_id": event_id}


def _find_free_slot(s, date, duration_minutes):
    """Deterministic: first slot in 09:00-18:00 that clears every booked event."""
    need = int(duration_minutes)
    busy = [(_mins(v["start"]), _mins(v["start"]) + v["duration"])
            for v in s["events"].values() if v["date"] == date and not v["cancelled"]]
    busy.sort()
    cursor = _mins("09:00")
    for start, end in busy:
        if start - cursor >= need:
            break
        cursor = max(cursor, end)
    if cursor + need > _mins("18:00"):
        return {"date": date, "found": False}
    return {"date": date, "found": True, "start_time": _hhmm(cursor),
            "duration_minutes": need}


def _mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(total):
    return f"{total // 60:02d}:{total % 60:02d}"


def _find_file(s, query):
    q = str(query).lower()
    hits = [{"file_id": k, **v} for k, v in s["files"].items()
            if not v["deleted"] and (q in v["name"].lower() or q in v["folder"].lower())]
    return {"query": query, "count": len(hits), "files": hits}


def _read_file(s, file_id):
    f = s["files"].get(file_id)
    if not f or f["deleted"]:
        return {"error": f"no such file: {file_id}"}
    return {"file_id": file_id, "name": f["name"], "content": f"contents of {f['name']}"}


def _rename_file(s, file_id, new_name):
    f = s["files"].get(file_id)
    if not f:
        return {"error": f"no such file: {file_id}"}
    f["name"] = new_name
    return {"ok": True, "file_id": file_id, "name": new_name}


def _move_file(s, file_id, folder):
    f = s["files"].get(file_id)
    if not f:
        return {"error": f"no such file: {file_id}"}
    if folder not in FOLDERS:
        return {"error": f"unknown folder: {folder}"}
    f["folder"] = folder
    return {"ok": True, "file_id": file_id, "folder": folder}


def _delete_file(s, file_id):
    f = s["files"].get(file_id)
    if not f:
        return {"error": f"no such file: {file_id}"}
    f["deleted"] = True
    return {"ok": True, "file_id": file_id}


def _share_file(s, file_id, recipient):
    f = s["files"].get(file_id)
    if not f:
        return {"error": f"no such file: {file_id}"}
    if recipient not in f["shared_with"]:
        f["shared_with"].append(recipient)
    return {"ok": True, "file_id": file_id, "shared_with": f["shared_with"]}


EXEC = {
    "find_events": _find_events, "get_event": _get_event, "create_event": _create_event,
    "move_event": _move_event, "cancel_event": _cancel_event,
    "find_free_slot": _find_free_slot, "find_file": _find_file, "read_file": _read_file,
    "rename_file": _rename_file, "move_file": _move_file, "delete_file": _delete_file,
    "share_file": _share_file,
}


def executors(profile) -> dict:
    return EXEC


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------

# Trigger regexes, declared by tool name rather than threaded through each
# call site. A matching trigger restricts decoding to the matched tools and
# forces a call, so the table is deliberately conservative: a trigger that
# fires on an under-specified request converts a correct refusal into a wrong
# call, which is exactly what the opt_triggers arm is there to measure.
TRIGGERS = {
    'find_events': [r'\\b(calendar|agenda|what .{0,15}(have|got) on)\\b'],
    'find_free_slot': [r'\\b(free (slot|hour|time)|fit .{0,12}(hour|meeting)|gap)\\b'],
    'find_file': [r'\\b(files?|documents?|notes)\\b'],
    'move_file': [r'\\bmove\\b.{0,25}\\b(file|folder|f_\\d+)\\b'],
    'move_event': [r'\\b(move|resched\\w*)\\b.{0,25}\\b(event|meeting|ev_\\d+)\\b'],
    'cancel_event': [r'\\b(cancel|scrap|drop)\\b.{0,25}\\b(event|meeting|ev_\\d+)\\b'],
    'delete_file': [r'\\bdelete\\b.{0,25}\\b(file|f_\\d+)\\b'],
    'share_file': [r'\\b(share|let .{0,15} see|give .{0,15} access)\\b'],
    'rename_file': [r'\\brename\\b'],
}


def _with_triggers(schemas, profile):
    if not profile.triggers:
        return schemas
    for schema in schemas:
        pats = TRIGGERS.get(schema["name"])
        if pats:
            schema["triggers"] = list(pats)
    return schemas


def tools(p) -> list[dict]:
    date = param(p, "string", "A date.",
                 "The calendar date as ISO 'YYYY-MM-DD'. Resolve 'today' and 'tomorrow' "
                 "against the date fact in the system turn.",
                 pattern=r"^\d{4}-\d{2}-\d{2}$")
    event_id = param(p, "string", "The event id.",
                     "The event identifier, of the form 'ev_123'. Obtain it from "
                     "find_events or get_event; never invent one.",
                     pattern=r"^ev_\d+$")
    file_id = param(p, "string", "The file id.",
                    "The file identifier, of the form 'f_123'. Obtain it from find_file; "
                    "never invent one.",
                    pattern=r"^f_\d+$")
    return _with_triggers([
        tool(p, "find_events", "Find events on a date.",
             "List the calendar events on one date. Use this first whenever the request "
             "names a meeting by description rather than by id.",
             {"date": date}, ["date"]),
        tool(p, "get_event", "Get one event.",
             "Read the full detail of a single calendar event by its id.",
             {"event_id": event_id}, ["event_id"]),
        tool(p, "create_event", "Create a calendar event.",
             "Add a new calendar event. Every field must come from the request: do not "
             "invent a title, a time, or a length.",
             {"title": param(p, "string", "The title.",
                             "The event title, copied from the request."),
              "date": date,
              "start_time": param(p, "string", "Start time.",
                                  "Start time as 24-hour 'HH:MM'.",
                                  pattern=r"^\d{2}:\d{2}$"),
              "duration_minutes": param(p, "integer", "Length in minutes.",
                                        "Length in whole minutes. One hour is 60.",
                                        bounds=(5, 480))},
             ["title", "date", "start_time", "duration_minutes"]),
        tool(p, "move_event", "Move an event to a new time.",
             "Reschedule an existing calendar event. Moves an event; never use this on "
             "files.",
             {"event_id": event_id, "new_date": date,
              "new_time": param(p, "string", "New time.", "New start time as 'HH:MM'.",
                                pattern=r"^\d{2}:\d{2}$")},
             ["event_id", "new_date", "new_time"]),
        tool(p, "cancel_event", "Cancel an event.",
             "Cancel a calendar event by id. Irreversible.",
             {"event_id": event_id}, ["event_id"]),
        tool(p, "find_free_slot", "Find a free slot on a date.",
             "Find the earliest gap of a given length in the working day on one date. "
             "Returns a start time; it does not book anything.",
             {"date": date,
              "duration_minutes": param(p, "integer", "Length in minutes.",
                                        "Required length in whole minutes.",
                                        bounds=(5, 480))},
             ["date", "duration_minutes"]),
        tool(p, "find_file", "Find files matching a query.",
             "Search documents by name or folder. Use this first whenever the request "
             "names a document in words rather than by id. Searches files, not calendar "
             "events.",
             {"query": param(p, "string", "Search text.",
                             "The words the user used to describe the document.")},
             ["query"]),
        tool(p, "read_file", "Read a file.",
             "Return the contents of one document by its id.",
             {"file_id": file_id}, ["file_id"]),
        tool(p, "rename_file", "Rename a file.",
             "Change a document's name. The new name must be stated in the request.",
             {"file_id": file_id,
              "new_name": param(p, "string", "The new name.",
                                "The new filename, copied from the request.")},
             ["file_id", "new_name"]),
        tool(p, "move_file", "Move a file to a folder.",
             "Move a document into a folder. Moves files; never use this on calendar "
             "events. Folders: finance, product, team, archive.",
             {"file_id": file_id,
              "folder": param(p, "string", "Destination folder.",
                              "The destination folder named in the request.",
                              enum=FOLDERS)},
             ["file_id", "folder"]),
        tool(p, "delete_file", "Delete a file.",
             "Permanently delete a document by id. Irreversible; only on an explicit "
             "instruction to delete.",
             {"file_id": file_id}, ["file_id"]),
        tool(p, "share_file", "Share a file with someone.",
             "Give another person access to a document. The recipient must be named in "
             "the request.",
             {"file_id": file_id,
              "recipient": param(p, "string", "Who to share with.",
                                 "The person named in the request.")},
             ["file_id", "recipient"]),
    ], p)


def translate_gold(calls: list[dict], profile) -> list[dict]:
    return [dict(c) for c in calls]
