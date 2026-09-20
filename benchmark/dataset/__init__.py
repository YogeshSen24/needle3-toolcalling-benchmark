"""Dataset assembly across the three environments."""
from __future__ import annotations


def build_all(environments: list[str] | None = None, levels: list[int] | None = None):
    from . import cases_business_ops, cases_smart_home, cases_workspace

    builders = {
        "smart_home": cases_smart_home.build_cases,
        "workspace": cases_workspace.build_cases,
        "business_ops": cases_business_ops.build_cases,
    }
    wanted = environments or list(builders)
    cases = []
    for name in wanted:
        cases.extend(builders[name]())
    if levels is not None:
        cases = [c for c in cases if c.complexity_level in levels]
    return cases
