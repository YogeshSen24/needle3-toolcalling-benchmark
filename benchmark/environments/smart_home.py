"""Tier 1 — smart home.

The easy end of the ladder on purpose: one entity type (a room), disjoint verbs,
small closed value sets, and no call that needs a value from a previous call
except in the conditional cases. If a tool-calling model cannot do this, nothing
further is worth measuring.
"""
from __future__ import annotations

from ._schema import apply_overrides, param, tool

ENV_ID = "smart_home"
TIER = 1

ROOMS = ["kitchen", "bedroom", "living_room", "study"]
COLORS = ["warm white", "cool white", "red", "green", "blue"]
PLAYLISTS = ["jazz", "focus", "rock", "classical"]
MIN_C, MAX_C = 10, 30

READ_ONLY = {"get_temperature", "get_thermostat"}


def initial_state(overrides: dict | None = None) -> dict:
    return apply_overrides({
        "lights": {r: {"on": False, "brightness": 0, "color": "warm white"} for r in ROOMS},
        "climate": {r: {"current_c": 21, "target_c": 21} for r in ROOMS},
        "blinds": {r: "closed" for r in ROOMS},
        "music": {r: {"playing": False, "playlist": None} for r in ROOMS},
    }, overrides)


# --------------------------------------------------------------------------
# Primitives — one implementation, shared by both granularities
# --------------------------------------------------------------------------

def _room(state, room):
    return None if room in state["lights"] else {"error": f"unknown room: {room}"}


def _light_on(s, room):
    if (e := _room(s, room)):
        return e
    light = s["lights"][room]
    light["on"] = True
    if light["brightness"] == 0:
        light["brightness"] = 100
    return {"ok": True, "room": room, **light}


def _light_off(s, room):
    if (e := _room(s, room)):
        return e
    s["lights"][room]["on"] = False
    return {"ok": True, "room": room, **s["lights"][room]}


def _light_brightness(s, room, percentage):
    if (e := _room(s, room)):
        return e
    try:
        percentage = int(percentage)
    except (TypeError, ValueError):
        return {"error": f"brightness not a number: {percentage}"}
    if not 0 <= percentage <= 100:
        return {"error": f"brightness out of range: {percentage}"}
    light = s["lights"][room]
    light["brightness"] = percentage
    light["on"] = percentage > 0
    return {"ok": True, "room": room, **light}


def _light_color(s, room, color):
    if (e := _room(s, room)):
        return e
    if color not in COLORS:
        return {"error": f"unknown color: {color}"}
    light = s["lights"][room]
    light["color"] = color
    light["on"] = True
    if light["brightness"] == 0:
        light["brightness"] = 100
    return {"ok": True, "room": room, **light}


def _set_temp(s, room, degrees_c):
    if (e := _room(s, room)):
        return e
    try:
        degrees_c = int(degrees_c)
    except (TypeError, ValueError):
        return {"error": f"temperature not a number: {degrees_c}"}
    if not MIN_C <= degrees_c <= MAX_C:
        return {"error": f"temperature out of range: {degrees_c}"}
    s["climate"][room]["target_c"] = degrees_c
    return {"ok": True, "room": room, "target_c": degrees_c}


def _get_temp(s, room):
    if (e := _room(s, room)):
        return e
    return {"room": room, "degrees_c": s["climate"][room]["current_c"]}


def _blinds(s, room, position):
    if (e := _room(s, room)):
        return e
    s["blinds"][room] = position
    return {"ok": True, "room": room, "blinds": position}


def _music_play(s, room, playlist):
    if (e := _room(s, room)):
        return e
    if playlist not in PLAYLISTS:
        return {"error": f"unknown playlist: {playlist}"}
    s["music"][room] = {"playing": True, "playlist": playlist}
    return {"ok": True, "room": room, **s["music"][room]}


def _music_stop(s, room):
    if (e := _room(s, room)):
        return e
    s["music"][room] = {"playing": False, "playlist": None}
    return {"ok": True, "room": room, **s["music"][room]}


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------

def _room_param(p):
    return param(
        p, "string",
        "The room.",
        "The room named in the request. One of: kitchen, bedroom, living_room, study.",
        enum=ROOMS)


def _verb_split(p) -> list[dict]:
    room = _room_param(p)
    return [
        tool(p, "turn_on_light",
             "Turn on the lights in a room.",
             "Switch a room's lights on. Use for 'turn on', 'switch on', 'lights on'. "
             "Never use this to change brightness or colour.",
             {"room": room}, ["room"],
             triggers=[r"\b(turn|switch|put|flick)\b.{0,20}\b(on)\b"]),
        tool(p, "turn_off_light",
             "Turn off the lights in a room.",
             "Switch a room's lights off. Use for 'turn off', 'switch off', 'kill the lights'. "
             "Never use this to dim; dimming is set_light_brightness.",
             {"room": room}, ["room"],
             triggers=[r"\b(turn|switch|put|kill|shut)\b.{0,20}\b(off|out)\b"]),
        tool(p, "set_light_brightness",
             "Set the brightness of the lights in a room.",
             "Set a room's light brightness to a percentage the user stated. Use for 'dim', "
             "'brighten', 'set to N percent'. The number comes from the request.",
             {"room": room,
              "percentage": param(p, "integer", "Brightness percentage.",
                                  "The brightness the user asked for, 0 to 100, as a whole "
                                  "number. Taken from the number in the request.",
                                  bounds=(0, 100))},
             ["room", "percentage"],
             triggers=[r"\b(dim|brighten|brightness)\b"]),
        tool(p, "set_light_color",
             "Set the colour of the lights in a room.",
             "Set a room's light colour to a colour the user named. Only for colours: "
             "warm white, cool white, red, green, blue.",
             {"room": room,
              "color": param(p, "string", "The colour.",
                             "The colour named in the request.", enum=COLORS)},
             ["room", "color"]),
        tool(p, "set_temperature",
             "Set the target temperature of a room.",
             "Set a room's target temperature in whole degrees Celsius, 10 to 30. "
             "Use for 'set to N degrees', 'make it N'. Never use this for lights.",
             {"room": room,
              "degrees_c": param(p, "integer", "Temperature in Celsius.",
                                 "The target temperature in whole degrees Celsius, from the "
                                 "number in the request.", bounds=(MIN_C, MAX_C))},
             ["room", "degrees_c"]),
        tool(p, "get_temperature",
             "Get the current temperature of a room.",
             "Read a room's current temperature in degrees Celsius. Use this first whenever "
             "the request depends on how warm or cold a room currently is.",
             {"room": room}, ["room"],
             triggers=[r"\b(how (warm|cold|hot)|current temperature|what.{0,10}temperature)\b"]),
        tool(p, "open_blinds", "Open the blinds in a room.",
             "Open a room's blinds. Use for 'open', 'raise', 'let the light in'.",
             {"room": room}, ["room"]),
        tool(p, "close_blinds", "Close the blinds in a room.",
             "Close a room's blinds. Use for 'close', 'shut', 'lower', 'blinds down'.",
             {"room": room}, ["room"]),
        tool(p, "play_music", "Play music in a room.",
             "Start a named playlist in a room. Only these playlists exist: jazz, focus, "
             "rock, classical.",
             {"room": room,
              "playlist": param(p, "string", "The playlist.",
                                "The playlist named in the request.", enum=PLAYLISTS)},
             ["room", "playlist"]),
        tool(p, "stop_music", "Stop the music in a room.",
             "Stop playback in a room. Use for 'stop', 'pause', 'kill the music'.",
             {"room": room}, ["room"]),
    ]


def _action_enum(p) -> list[dict]:
    room = _room_param(p)
    return [
        tool(p, "control_lights",
             "Control the lights in a room.",
             "Switch a room's lights on or off, dim them to a percentage, or set their "
             "colour. action='dim' requires brightness_percent; action='color' requires "
             "color. Setting a brightness also turns the lights on.",
             {"room": room,
              "action": param(p, "string", "What to do.",
                              "on, off, dim, or color.", enum=["on", "off", "dim", "color"]),
              "brightness_percent": param(p, "integer", "Brightness percentage.",
                                          "Brightness 0 to 100, for action='dim'.",
                                          bounds=(0, 100)),
              "color": param(p, "string", "The colour.",
                             "The colour, for action='color'.", enum=COLORS)},
             ["room", "action"],
             triggers=[r"\b(light|lights|lamp|dim|brighten)\b"]),
        tool(p, "set_thermostat",
             "Set the target temperature of a room.",
             "Set a room's target temperature in whole degrees Celsius, 10 to 30.",
             {"room": room,
              "temperature": param(p, "integer", "Temperature in Celsius.",
                                   "Target temperature in whole degrees Celsius.",
                                   bounds=(MIN_C, MAX_C))},
             ["room", "temperature"]),
        tool(p, "get_thermostat",
             "Get the current temperature of a room.",
             "Read a room's current temperature in degrees Celsius. Use this first whenever "
             "the request depends on how warm or cold a room currently is.",
             {"room": room}, ["room"],
             triggers=[r"\b(how (warm|cold|hot)|current temperature)\b"]),
        tool(p, "control_blinds", "Open or close the blinds in a room.",
             "Open or close a room's blinds.",
             {"room": room,
              "action": param(p, "string", "open or close.", "open or close.",
                              enum=["open", "close"])},
             ["room", "action"]),
        tool(p, "control_music", "Play or stop music in a room.",
             "Start or stop playback in a room. action='play' requires a playlist from: "
             "jazz, focus, rock, classical.",
             {"room": room,
              "action": param(p, "string", "play or stop.", "play or stop.",
                              enum=["play", "stop"]),
              "playlist": param(p, "string", "The playlist.",
                                "The playlist, for action='play'.", enum=PLAYLISTS)},
             ["room", "action"]),
    ]


def tools(profile) -> list[dict]:
    return _action_enum(profile) if profile.granularity == "action_enum" else _verb_split(profile)


def _control_lights(s, room, action, brightness_percent=None, color=None):
    if action == "on":
        # A supplied brightness is honoured: a device API that silently dropped
        # it would be the thing at fault, not the model.
        if brightness_percent is not None:
            return _light_brightness(s, room, brightness_percent)
        return _light_on(s, room)
    if action == "off":
        return _light_off(s, room)
    if action == "dim":
        if brightness_percent is None:
            return {"error": "dim requires brightness_percent"}
        return _light_brightness(s, room, brightness_percent)
    if action == "color":
        if color is None:
            return {"error": "color action requires color"}
        return _light_color(s, room, color)
    return {"error": f"unknown action: {action}"}


def _control_music(s, room, action, playlist=None):
    if action == "play":
        return _music_play(s, room, playlist) if playlist else {"error": "play requires playlist"}
    if action == "stop":
        return _music_stop(s, room)
    return {"error": f"unknown action: {action}"}


_VERB_EXEC = {
    "turn_on_light": _light_on,
    "turn_off_light": _light_off,
    "set_light_brightness": _light_brightness,
    "set_light_color": _light_color,
    "set_temperature": _set_temp,
    "get_temperature": _get_temp,
    "open_blinds": lambda s, room: _blinds(s, room, "open"),
    "close_blinds": lambda s, room: _blinds(s, room, "closed"),
    "play_music": _music_play,
    "stop_music": _music_stop,
}

_ENUM_EXEC = {
    "control_lights": _control_lights,
    "set_thermostat": lambda s, room, temperature: _set_temp(s, room, temperature),
    "get_thermostat": _get_temp,
    "control_blinds": lambda s, room, action: _blinds(s, room, "open" if action == "open" else "closed"),
    "control_music": _control_music,
}


def executors(profile) -> dict:
    return _ENUM_EXEC if profile.granularity == "action_enum" else _VERB_EXEC


_TO_ENUM = {
    "turn_on_light": lambda a: ("control_lights", {"room": a["room"], "action": "on"}),
    "turn_off_light": lambda a: ("control_lights", {"room": a["room"], "action": "off"}),
    "set_light_brightness": lambda a: ("control_lights", {
        "room": a["room"], "action": "dim", "brightness_percent": a["percentage"]}),
    "set_light_color": lambda a: ("control_lights", {
        "room": a["room"], "action": "color", "color": a["color"]}),
    "set_temperature": lambda a: ("set_thermostat", {
        "room": a["room"], "temperature": a["degrees_c"]}),
    "get_temperature": lambda a: ("get_thermostat", {"room": a["room"]}),
    "open_blinds": lambda a: ("control_blinds", {"room": a["room"], "action": "open"}),
    "close_blinds": lambda a: ("control_blinds", {"room": a["room"], "action": "close"}),
    "play_music": lambda a: ("control_music", {
        "room": a["room"], "action": "play", "playlist": a["playlist"]}),
    "stop_music": lambda a: ("control_music", {"room": a["room"], "action": "stop"}),
}


def translate_gold(calls: list[dict], profile) -> list[dict]:
    """Gold is authored once in verb_split form; the enum form is derived.

    Hand-maintaining two parallel gold sets would let them drift, and the drift
    would show up as a granularity effect that is really an authoring bug.
    """
    if profile.granularity != "action_enum":
        return [dict(c) for c in calls]
    out = []
    for call in calls:
        fn = _TO_ENUM.get(call["name"])
        if fn is None:
            raise ValueError(f"no action_enum translation for {call['name']}")
        name, args = fn(call.get("arguments") or {})
        out.append({"name": name, "arguments": args})
    return out
