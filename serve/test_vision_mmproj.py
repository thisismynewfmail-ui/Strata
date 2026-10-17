"""The image encoder next to the model's files at the server's start (serve/mmproj.py, vision_from_mmproj): a config
without a "vision" section - a file put next to the model after setup, started with run-<model>.bat - gets images on
for that start, with the strata-vision beside the engine, --vision and the GPU encoder's VRAM reserve; --mmproj FILE
(which run-<model>.bat passes) does the same; a config's own section, a declined file, an engine without an encoder
and a file for another model are left as they were, with a line saying why.  The encoder files are real (tiny) GGUFs.

    python -m serve.test_vision_mmproj
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import serve.server as server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gguf_writer import GGUFWriter  # noqa: E402

VEXE = "strata-vision.exe" if server.os.name == "nt" else "strata-vision"


def encoder(path: Path, width: int = 2560) -> Path:
    w = GGUFWriter()
    w.add("general.architecture", "clip")
    w.add("clip.has_vision_encoder", True)
    w.add("clip.vision.projection_dim", width, "u32")
    w.add_f32("mm.2.weight", np.zeros((4, 64), dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    return w.write(path)


class FromMmproj(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.eng = t / "engine"
        self.eng.mkdir()
        (self.eng / "strata").write_bytes(b"")
        (self.eng / VEXE).write_bytes(b"")
        self.models = t / "models" / "swift-IQ3_S"
        self.models.mkdir(parents=True)
        self.shard = self.models / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
        self.shard.write_bytes(b"")

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, **kw):
        return {"exe": str(self.eng / "strata"), "args": ["--pack", "p", "--native", str(self.shard),
                                                          "--max-context", "32768"], **kw}

    def test_found_next_to_the_model(self):
        mm = encoder(self.models / "mmproj-Swift-Qwen3.8-Flash-Next-GSQ-RCO.gguf")
        cfg = self.cfg()
        note, added = server.vision_from_mmproj(cfg)
        self.assertIn("images: ON", note)
        self.assertIn(str(mm), note)
        self.assertEqual(cfg["vision"], {"exe": str(self.eng / VEXE), "mmproj": str(mm), "model": str(self.shard),
                                         "gpu": True, "max_tokens": 1024})
        self.assertEqual(added, ["--vision", "--vram-reserve-mib", "700"])
        self.assertEqual(cfg["args"][-3:], added)
        # what the engine is started with: no "added --vision" note (#1322) on top
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a = server.engine_args(cfg)
        self.assertEqual(a.count("--vision"), 1)
        self.assertEqual(out.getvalue(), "")
        # and the encoder's own command line
        v = server.Vision(cfg["vision"], lazy=True)
        self.assertEqual(v.spawn[0][1:], ["--mmproj", str(mm), "--model", str(self.shard), "--gpu", "--max-tokens", "1024"])

    def test_a_reserve_of_the_users_own_is_kept(self):
        encoder(self.models / "mmproj-x.gguf")
        cfg = self.cfg()
        cfg["args"] += ["--vram-reserve-mib", "2048"]
        server.vision_from_mmproj(cfg)
        self.assertEqual(cfg["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(cfg["args"][cfg["args"].index("--vram-reserve-mib") + 1], "2048")

    def test_amd_reads_on_the_cpu(self):
        encoder(self.models / "mmproj-x.gguf")
        cfg = self.cfg(backend="hip")
        with mock.patch.object(server.os, "name", "posix"):
            note, added = server.vision_from_mmproj(cfg)
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertEqual(cfg["vision"]["max_tokens"], 300)
        self.assertIn("threads", cfg["vision"])
        self.assertEqual(added, ["--vision"])

    def test_left_as_it_was(self):
        mm = encoder(self.models / "mmproj-x.gguf")
        own = {"exe": "v", "mmproj": "/mine/mmproj-Q8_0.gguf", "gpu": False}
        cfg = self.cfg(vision=dict(own))
        before = json.dumps(cfg)
        self.assertEqual(server.vision_from_mmproj(cfg), (None, []))       # the caller says "images: on"
        self.assertEqual(server.vision_from_mmproj(cfg, str(mm)), (None, []))   # the config wins over --mmproj
        self.assertEqual(json.dumps(cfg), before)
        cfg = self.cfg(vision_declined=str(mm))                             # left off on purpose: said, not changed
        before = json.dumps(cfg)
        note, added = server.vision_from_mmproj(cfg)
        self.assertIn("images: off - setup left them off", note)
        self.assertEqual((json.dumps(cfg), added), (before, []))

    def test_nothing_there(self):
        cfg = self.cfg()
        note, added = server.vision_from_mmproj(cfg)
        self.assertEqual(note, f"[strata] images: off - no image encoder (mmproj*.gguf) next to the model in "
                               f"{self.models}")
        self.assertEqual(added, [])
        self.assertNotIn("vision", cfg)

    def test_says_why_not(self):
        encoder(self.models / "mmproj-other.gguf", width=4096)
        cfg = self.cfg()
        note, _ = server.vision_from_mmproj(cfg)
        self.assertIn("images: off", note)
        self.assertIn("made for another model", note)
        self.assertNotIn("vision", cfg)
        (self.models / "mmproj-other.gguf").unlink()
        encoder(self.models / "mmproj-x.gguf")
        (self.eng / VEXE).unlink()
        note, _ = server.vision_from_mmproj(cfg)
        self.assertIn("is not installed", note)
        self.assertIn("START-HERE.bat", note)
        self.assertNotIn("vision", cfg)
        for backend in ("sycl",):
            note, _ = server.vision_from_mmproj(self.cfg(backend=backend))
            self.assertIn("no image encoder", note)

    def test_mmproj_flag(self):
        mm = encoder(Path(self.tmp.name) / "elsewhere" / "mmproj-Q8_0.gguf")
        cfg = self.cfg()
        note, _ = server.vision_from_mmproj(cfg, str(mm))
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertIn("--mmproj", note)
        note, _ = server.vision_from_mmproj(self.cfg(), str(Path(self.tmp.name) / "missing.gguf"))
        self.assertIn("there is no such file", note)


class RefusalSaysWhy(unittest.TestCase):
    def test_the_400_names_the_reason(self):
        svc = server.Service.__new__(server.Service)
        svc.vision = None
        svc.tok = Tok()
        svc.embeddings = SimpleNamespace(path=None, keys=None)
        svc._note_unreadable_tool_images = lambda messages, cancel: None
        svc.encode_prompt = lambda messages, tools, kwargs: [0]
        msgs = [{"role": "user", "content": [{"type": "image", "source": "data:image/png;base64,AA=="},   # normalized
                                             {"type": "text", "text": "what is this?"}]}]
        for why, want in ((None, "put the model's mmproj .gguf next to its model files"),
                          ("no image encoder (mmproj*.gguf) next to the model in D:\\m", "next to the model in D:\\m")):
            svc.vision_off_why = why
            with self.assertRaises(ValueError) as e:
                svc.prepare(msgs, None, {})
            self.assertIn("started without the vision encoder", str(e.exception))
            self.assertIn(want, str(e.exception))


class Tok:
    """strata_tokenizer.Tokenizer: these starts never tokenize (the engine stops them first)."""
    def __init__(self, *a):
        pass

    def encode(self, text, parse_special=False):
        return [0]


class Start(unittest.TestCase):
    """server.main(): the encoder found at the start reaches the Vision and the engine; one that does not start
    leaves the model to start without images."""

    def run_main(self, vision_side_effect=None):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            tk = t / "tokenizer"
            tk.mkdir()
            (tk / "vocab.json").write_text(json.dumps({"a": 0}))
            (tk / "merges.txt").write_text("")
            (tk / "token_type.json").write_text("[1]")
            (t / "engine").mkdir()
            (t / "engine" / "strata").write_bytes(b"")
            (t / "engine" / VEXE).write_bytes(b"")
            shard = t / "models" / "m" / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
            mm = encoder(shard.parent / "mmproj-Swift-Qwen3.8-Flash-Next-GSQ-RCO.gguf")
            shard.write_bytes(b"")
            cfg = t / "strata-swift-iq3_s.json"
            cfg.write_text(json.dumps({"exe": str(t / "engine" / "strata"), "tokenizer": str(tk), "cwd": d,
                                       "args": ["--native", str(shard), "--max-context", "8192"]}))
            seen = {}

            class Stop(Exception):
                pass

            def engine(exe, args, **kw):
                seen["args"] = args
                raise Stop

            def vision(vcfg, **kw):
                seen["vision"] = vcfg
                if vision_side_effect:
                    raise vision_side_effect
                return object()
            out = io.StringIO()
            with mock.patch.object(server, "StrataEngine", engine), mock.patch.object(server, "Vision", vision), \
                    mock.patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(Tokenizer=Tok)}), \
                    mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(cfg),
                                                    "--port", "0"]), contextlib.redirect_stdout(out):
                with self.assertRaises(Stop):
                    server.main()
            return seen, out.getvalue(), mm

    def test_images_on(self):
        seen, out, mm = self.run_main()
        self.assertEqual(seen["vision"]["mmproj"], str(mm))
        self.assertIn("--vision", seen["args"])
        self.assertIn("images: ON - image encoder", out)

    def test_an_encoder_that_does_not_start(self):
        seen, out, mm = self.run_main(RuntimeError("the vision encoder did not start: no CUDA"))
        self.assertNotIn("--vision", seen["args"])
        self.assertNotIn("--vram-reserve-mib", seen["args"])
        self.assertIn("images: off - the vision encoder did not start", out)


if __name__ == "__main__":
    unittest.main()
