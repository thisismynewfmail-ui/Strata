"""Tests for the image encoder setup finds next to the model's files (an mmproj*.gguf): it is used instead of the
download and images are on without the question; --vision no leaves it unused (and START-HERE does not ask again);
a file for another model or a cut-off one is named and not used; --mmproj FILE gives one by hand; and START-HERE on a
text-only model whose folder now has one sets that model up again with images.  Every outside effect is mocked
(tools/test_setup_golden.py's install()); the encoder files are real (tiny) GGUFs.

    python tools/test_setup_mmproj.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import unittest.mock as mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import gguf_reader  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from test_setup_golden import PROFILES, FakeGGUF, install  # noqa: E402

FAM = setup.FAMILIES["qwen"]
TAG = FAM["tag"] + "Q2_0"


def encoder(path: Path, width: int = setup.FLASH_NEXT_EMBD, arch: str = "clip", cut: bool = False) -> Path:
    """A tiny mmproj GGUF: the keys strata-vision reads, one tensor."""
    w = GGUFWriter()
    w.add("general.architecture", arch)
    w.add("clip.has_vision_encoder", True)
    w.add("clip.vision.projection_dim", width, "u32")
    w.add_f32("mm.2.weight", np.zeros((4, 64), dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    w.write(path)
    if cut:
        path.write_bytes(path.read_bytes()[:-200])
    return path


class Reader:
    """gguf_reader for install(): the real reader for the encoders, the harness's fake for the model's shards."""
    @staticmethod
    def GGUFFile(p):
        return gguf_reader.GGUFFile(p) if setup.mmproj_name(Path(p).name) else FakeGGUF(p)


READER = mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=Reader.GGUFFile)})
REAL_RUN_SCRIPT = setup.write_run_script              # install() replaces it


class Install(unittest.TestCase):
    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"     # --models-dir: the model's folder is <models>/<tag>
        self.folder = self.models / TAG

    def tearDown(self):
        self.tmp.cleanup()

    def run_setup(self, *extra, answers=None, configs=(), patches=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--context", "32768",
                                              "--no-start", "--models-dir", str(self.models), *extra],
                       answers=answers, extra=[READER, *patches], configs=configs)

    def test_found_next_to_the_model_turns_images_on(self):
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(code, 0, out)
        self.assertIn(f"image encoder found next to the model: {mm}", out)
        self.assertIn("images: on", out)
        v = cfg["vision"]
        self.assertEqual(v["mmproj"], str(mm))
        self.assertTrue(v["gpu"])
        self.assertEqual(v["max_tokens"], setup.VISION["gpu"]["max_tokens"])
        self.assertEqual(v["exe"], "<T>/engine/" + setup.VEXE)
        self.assertEqual(Path(v["model"]).parent, self.folder)       # the text model beside it, vocab-only
        a = cfg["args"]
        self.assertIn("--vision", a)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], str(setup.VISION["gpu"]["reserve_mib"]))
        self.assertNotIn("vision_declined", cfg)
        self.assertIn("found next to the model, nothing downloaded", out)
        self.assertFalse((self.models / setup.MMPROJ).exists())       # the encoder was not downloaded
        self.assertIn("Images:", out)                                  # the closing summary says how to send them

    def test_any_mmproj_name_and_no_question(self):
        mm = encoder(self.folder / "Qwen3.8-Flash-Next-mmproj-F16.gguf")
        code, out, cfg, asked = self.run_setup(answers="")             # Enter for every question
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertFalse(any("images" in q.lower() for q in asked), asked)

    def test_vision_no_leaves_it_unused(self):
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, cfg, _ = self.run_setup("--vision", "no")
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])
        self.assertEqual(cfg["vision_declined"], str(mm))              # START-HERE does not set it up again
        self.assertIn("is not used (--vision no)", out)

    def test_vision_cpu_uses_the_found_file_on_the_cpu(self):
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, cfg, _ = self.run_setup("--vision", "cpu")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertIn("threads", cfg["vision"])

    def test_an_encoder_for_another_model_is_not_used(self):
        encoder(self.folder / "mmproj-other-model-F16.gguf", width=4096)
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(code, 0, out)
        self.assertIn("made for another model (its projection is 4096 wide", out)
        self.assertNotIn("vision", cfg)                                 # as without the file: the default "no"
        self.assertNotIn("vision_declined", cfg)

    def test_a_cut_off_encoder_is_not_used(self):
        encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf", cut=True)
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(code, 0, out)
        self.assertIn("it is cut off", out)
        self.assertNotIn("vision", cfg)

    def test_the_familys_own_file_first_and_the_others_named(self):
        encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf")
        own = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(cfg["vision"]["mmproj"], str(own))
        self.assertIn("2 image encoders are next to the model", out)
        self.assertIn("mmproj-Qwen3.8-Flash-Next-Q8_0.gguf", out)

    def test_mmproj_flag(self):
        mm = encoder(Path(self.tmp.name) / "mine" / "encoder-q8.mmproj.gguf")
        code, out, cfg, _ = self.run_setup("--mmproj", str(mm))
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertIn("(--mmproj, nothing downloaded)", out)

    def test_mmproj_flag_refuses_a_wrong_file(self):
        bad = encoder(Path(self.tmp.name) / "mine" / "mmproj-other.gguf", width=3584)
        with contextlib.redirect_stderr(io.StringIO()):
            code, out, cfg, _ = self.run_setup("--mmproj", str(bad))
        self.assertNotEqual(code, 0)
        self.assertIn("made for another model", out)
        code, out, cfg, _ = self.run_setup("--mmproj", str(Path(self.tmp.name) / "missing.gguf"))
        self.assertNotEqual(code, 0)
        self.assertIn("there is no such file", out)

    def test_found_encoder_wins_over_an_earlier_hand_set_one(self):
        other = encoder(Path(self.tmp.name) / "elsewhere" / "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf")
        old = {"exe": "x", "args": [], "vision": {"exe": "v", "mmproj": str(other), "gpu": True, "max_tokens": 1024}}
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, cfg, _ = self.run_setup(configs=[(f"strata-{TAG.lower()}.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))

    def test_start_here_sets_a_text_only_model_up_again_with_images(self):
        code, out, first, _ = self.run_setup("--context", "65536")
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", first)
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        started = mock.Mock(return_value=0)
        # a plain start (START-HERE.bat with no options) on the installed text-only model
        code, out, cfg, _ = install(self.ram, self.found, ["--models-dir", str(self.models)], extra=[
            READER, mock.patch.object(setup, "start", started)], configs=[(f"strata-{TAG.lower()}.json", first)])
        self.assertEqual(code, 0, out)
        self.assertIn(f"image encoder found next to the model: {mm}", out)
        self.assertIn("up again with images", out)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertIn("--vision", cfg["args"])
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "65536")      # the other settings are the earlier ones
        self.assertEqual(started.call_count, 1)                         # and then the model starts

    def test_start_here_keeps_the_model_files_where_they_are(self):
        # set up with --gguf-dir (a folder setup does not look in by itself), then started without it
        own = Path(self.tmp.name) / "my-ggufs"
        own.mkdir()
        for i in (1, 2):
            shard = own / setup.model_file(FAM, "Q2_0", i)
            shard.write_bytes(b"")
            setup.mark(shard)
        code, out, first, _ = self.run_setup("--gguf-dir", str(own))
        self.assertEqual(code, 0, out)
        mm = encoder(own / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        got = []
        code, out, cfg, _ = install(self.ram, self.found, [], extra=[
            READER, mock.patch.object(setup, "start", mock.Mock(return_value=0)),
            mock.patch.object(setup, "download", lambda url, dst, what=None: got.append(dst))],
            configs=[(f"strata-{TAG.lower()}.json", first)])
        self.assertEqual(code, 0, out)
        self.assertEqual(got, [])                                       # nothing downloaded
        a = cfg["args"]
        self.assertEqual(Path(a[a.index("--native") + 1]).parent, own)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))

    def test_setup_bat_swift_iq3_s(self):
        # SETUP.bat (START-HERE.bat --setup): Swift 1.5, IQ3_S from the menus, the encoder next to the model; the
        # real run-swift-iq3_s.bat is written
        swift = setup.FAMILIES["swift"]
        folder = self.models / (swift["tag"] + "IQ3_S")
        mm = encoder(folder / "mmproj-Swift-Qwen3.8-Flash-Next-GSQ-RCO.gguf")
        written = {}

        def run_script(*a, **k):
            p = REAL_RUN_SCRIPT(*a, **k)
            written[p.name] = p.read_bytes().decode("utf-8")
            return p
        names = [m for m in setup.MODELS if "swift" in setup.MODELS[m].get("families", ("qwen", "swift"))]
        names.sort(key=lambda m: bool(setup.MODELS[m].get("experimental")))
        code, out, cfg, asked = install(self.ram, self.found, ["--setup", "--no-start", "--models-dir", str(self.models)],
                                        answers={"Which model?": "2", "Which size?": str(names.index("IQ3_S") + 1)},
                                        extra=[READER, mock.patch.object(setup, "WIN", True),
                                               mock.patch.object(setup, "write_run_script", run_script)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "swift-1.5-iq3_s")
        self.assertFalse(any("images" in q.lower() for q in asked), asked)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertTrue(cfg["vision"]["gpu"])
        self.assertIn("--vision", cfg["args"])
        bat = written["run-swift-iq3_s.bat"]
        self.assertIn(f'"--mmproj" "{mm}"', bat)
        self.assertIn(f"rem Images: on - image encoder {mm}\r\n", bat)
        self.assertIn("start script: run-swift-iq3_s.bat", out)

    def test_start_here_just_starts_when_images_were_declined(self):
        mm = encoder(self.folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        code, out, first, _ = self.run_setup("--vision", "no")
        self.assertEqual(first["vision_declined"], str(mm))
        started = mock.Mock(return_value=0)
        code, out, cfg, _ = install(self.ram, self.found, ["--models-dir", str(self.models)], extra=[
            READER, mock.patch.object(setup, "start", started)], configs=[(f"strata-{TAG.lower()}.json", first)])
        self.assertEqual(code, 0, out)
        self.assertNotIn("up again with images", out)
        self.assertEqual(started.call_count, 1)


class AMD(unittest.TestCase):
    """An AMD card: the found encoder runs on the CPU on Linux; the ready-made Windows AMD engine has none (off, and
    remembered so START-HERE does not set the model up again)."""
    CARD = {"index": 0, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 23.98, "arch": "gfx1100"}

    @staticmethod
    def engine_step(a, gpu, hip, amd, cuda_tk, vision):
        eng = setup.ROOT / "engine"
        return Path("llama"), eng, json.loads((eng / "BUILD.json").read_text()), [], gpu, vision

    def run_setup(self, win):
        with tempfile.TemporaryDirectory() as d:
            models = Path(d) / "models"
            mm = encoder(models / TAG / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
            code, out, cfg, _ = install(64, [], ["--family", "qwen", "--model", "Q2_0", "--context", "32768",
                                                 "--no-start", "--models-dir", str(models)], extra=[
                READER, mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(self.CARD)]),
                mock.patch.object(setup, "WIN", win), mock.patch.object(setup, "engine_step", self.engine_step),
                mock.patch.object(setup, "amd_device_access_problem", lambda: None),
                mock.patch.object(setup, "hipblaslt_table", lambda *a, **k: None),
                mock.patch.object(setup, "linux_desktop", lambda: False)])
            self.assertEqual(code, 0, out)
            return mm, out, cfg

    def test_linux_cpu_encoder(self):
        mm, out, cfg = self.run_setup(False)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertIn("images: on (encoder on the CPU)", out)

    def test_windows_off(self):
        mm, out, cfg = self.run_setup(True)
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg["vision_declined"], str(mm))
        self.assertIn("Linux-only for now", out)


class Helpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_names(self):
        for n in ("mmproj-F16.gguf", "Qwen-MMPROJ-bf16.gguf", "model.mmproj"):
            self.assertTrue(setup.mmproj_name(n), n)
        for n in ("mmproj-F16.gguf.part", "mmproj-F16.gguf.done", "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"):
            self.assertFalse(setup.mmproj_name(n), n)

    def test_problems(self):
        self.assertIsNone(setup.mmproj_problem(encoder(self.t / "a" / "mmproj.gguf")))
        self.assertIn("not an image encoder", setup.mmproj_problem(encoder(self.t / "b" / "mmproj.gguf", arch="llama")))
        junk = self.t / "mmproj-junk.gguf"
        junk.write_bytes(b"not a gguf at all")
        self.assertIn("not a GGUF file", setup.mmproj_problem(junk))

    def test_unused_mmproj(self):
        folder = self.t / "models" / TAG
        shard = folder / setup.model_file(FAM, "Q2_0", 1)
        folder.mkdir(parents=True)
        shard.write_bytes(b"")
        cfg = {"exe": "x", "args": ["--pack", "p", "--native", str(shard), "--max-context", "32768"]}
        path = self.t / f"strata-{TAG.lower()}.json"

        def check(c):
            path.write_text(json.dumps(c))
            return setup.unused_mmproj(path)
        self.assertIsNone(check(cfg))                                   # no encoder there
        mm = encoder(folder / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        self.assertEqual(check(cfg), mm)
        self.assertIsNone(check({**cfg, "vision": {"mmproj": str(mm)}}))      # already has images
        self.assertIsNone(check({**cfg, "vision_declined": str(mm)}))         # left off on purpose
        self.assertIsNone(check({**cfg, "backend": "sycl"}))                  # no encoder on Intel Arc
        with mock.patch.object(setup, "WIN", True):
            self.assertIsNone(check({**cfg, "backend": "hip"}))               # nor in the Windows AMD engine

    def test_run_script_names_the_encoder(self):
        mm = str(self.t / "models" / "swift-IQ3_S" / "mmproj-Swift-Qwen3.8-Flash-Next-GSQ-RCO.gguf")
        cfg = self.t / "strata-swift-iq3_s.json"
        for win, name in ((True, "run-swift-iq3_s.bat"), (False, "run-swift-iq3_s.sh")):
            with mock.patch.object(setup, "ROOT", self.t), mock.patch.object(setup, "WIN", win):
                cfg.write_text(json.dumps({"exe": "x", "args": [], "vision": {"mmproj": mm, "gpu": True}}))
                text = setup.write_run_script("swift-IQ3_S", cfg, 8080).read_bytes().decode("utf-8")
                self.assertEqual((self.t / name).read_bytes().decode("utf-8"), text)
                self.assertIn(f'"--mmproj" "{mm}"', text)
                self.assertIn(f"Images: on - image encoder {mm}", text)
                self.assertIn(f'"--config" "{cfg}" "--port" "8080" "--open" "--mmproj"', text)
                # a text-only model's script is as before
                cfg.write_text(json.dumps({"exe": "x", "args": []}))
                text = setup.write_run_script("swift-IQ3_S", cfg, 8080).read_bytes().decode("utf-8")
                self.assertNotIn("mmproj", text)
                self.assertNotIn("Images", text)
                self.assertTrue(text.endswith(('"--open"\r\nif errorlevel 1 pause\r\n' if win else '"--open"\n')))

    def test_carry_over_keep_mmproj(self):
        mine = encoder(self.t / "mmproj-Q8_0.gguf")
        old = {"vision": {"mmproj": str(mine), "gpu": True}}
        new = {"vision": {"mmproj": "/data/mmproj-BF16.gguf", "gpu": True}}
        setup.carry_over(old, new)
        self.assertEqual(new["vision"]["mmproj"], str(mine))           # #625: kept over setup's download
        new = {"vision": {"mmproj": "/models/found/mmproj-BF16.gguf", "gpu": True}}
        setup.carry_over(old, new, keep_mmproj=False)
        self.assertEqual(new["vision"]["mmproj"], "/models/found/mmproj-BF16.gguf")   # not over the user's new one


if __name__ == "__main__":
    unittest.main()
