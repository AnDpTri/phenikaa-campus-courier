"""Natural-language workstream: mission text to structured Mission.

Layout:
  text.py      accent folding, tokenisation, one-edit typo repair
  lexicon.py   hand-written place names, cue phrases, urgency/fragility phrases
  parser.py    text -> ParsedMission (goal/via TargetSpec, urgent, fragile); never sees the map
  resolver.py  ParsedMission + landmarks (from CV) -> courier.common.Mission
"""

from .parser import MissionParser, ParsedMission, TargetSpec, parse_mission
from .resolver import resolve, resolve_target

__all__ = [
    "MissionParser",
    "ParsedMission",
    "TargetSpec",
    "parse_mission",
    "resolve",
    "resolve_target",
]
