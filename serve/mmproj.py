"""The image encoder next to a model's files: an mmproj*.gguf, as the GGUF repositories ship it.

Setup uses one it finds instead of downloading the encoder and writes it into the run config; the server turns images
on with one for a start whose config has none (a file put there after setup, started with run-<model>.bat).  Both
check it the same way, from its headers only: a CLIP vision encoder whose projection is as wide as the model's
embeddings (strata-vision reads that width from clip.vision.projection_dim and refuses a file without it, and the
engine refuses image embeddings of any other width), and not cut off.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every Flash-Next family's text model is 2560 wide (qwen4exp.embedding_length)
FLASH_NEXT_EMBD = 2560


def mmproj_name(name: str) -> bool:
    """A file name an image encoder goes by: "mmproj" in it, a .gguf (or .mmproj) file."""
    n = name.lower()
    return "mmproj" in n and n.endswith((".gguf", ".mmproj"))


def mmproj_problem(p: Path, width: int = FLASH_NEXT_EMBD) -> str | None:
    """Why `p` cannot be this model's image encoder (not a GGUF, no vision encoder in it, made for a model of another
    width, cut off), from its headers only; None when strata-vision can use it."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(p)
    except (OSError, ValueError, KeyError, struct.error) as e:
        return f"not a GGUF file that can be read ({e})"
    md = getattr(g, "metadata", None) or {}
    if md.get("general.architecture") != "clip" or md.get("clip.has_vision_encoder") is False:
        return "it is not an image encoder"
    got = md.get("clip.vision.projection_dim")
    if not isinstance(got, int):
        return "it has no clip.vision.projection_dim, which strata-vision needs"
    if got != width:
        return f"it is made for another model (its projection is {got} wide, this model's is {width})"
    if p.stat().st_size < g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0):
        return "it is cut off (shorter than its own tensor directory says): a download that did not finish?"
    return None


def scan(folders, prefer: str | None = None) -> tuple[list[Path], list[tuple[Path, str]]]:
    """The usable image encoders of the first of `folders` that has one (absolute paths; the file named `prefer`
    first, then by name), and every file passed over on the way, with the reason."""
    passed = []
    for folder in dict.fromkeys(Path(f) for f in folders):
        try:
            cands = sorted((p for p in folder.iterdir() if mmproj_name(p.name) and p.is_file()),
                           key=lambda p: (p.name != prefer, p.name.lower()))
        except OSError:                                # no such folder (yet)
            continue
        good = []
        for p in cands:
            why = mmproj_problem(p)
            if why is None:
                good.append(Path(os.path.abspath(p)))
            else:
                passed.append((p, why))
        if good:
            return good, passed
    return [], passed
