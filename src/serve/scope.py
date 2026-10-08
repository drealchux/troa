"""
Resolve the router's loose document scope onto real document names.

The router returns names as a user would write them ("W-10", "P-5",
"gsa020k_gas_masters"), while stored documents are file stems such as
"ola001k_oil_well_status_w10". Both sides are normalised to lowercase
alphanumerics and matched by substring, so "W-10" -> "w10" matches
"ola001koilwellstatusw10".
"""

from __future__ import annotations

import re
from typing import Iterable, Optional


def _norm(name: str) -> str:
    name = name.lower()
    if name.endswith(".pdf"):
        name = name[:-4]
    return re.sub(r"[^a-z0-9]", "", name)


def resolve_scope(scope: Optional[Iterable[str]], available: Iterable[str]) -> list[str]:
    """Return the entries of `available` matched by any name in `scope`, in sorted order.

    Names shorter than 2 characters after normalisation are ignored. An empty
    result means "no usable scope": callers should search the whole corpus.
    """
    available = list(available)
    hits: set[str] = set()
    for name in scope or []:
        n = _norm(str(name))
        if len(n) < 2:
            continue
        for doc in available:
            d = _norm(doc)
            if n in d or d in n:
                hits.add(doc)
    return sorted(hits)
