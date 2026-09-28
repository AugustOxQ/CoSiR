"""Structural checks for the ArtELingo retrieval annotation converter."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/prepare_artelingo_retrieval_eval.py"


def load_converter():
    assert SCRIPT.is_file(), f"Missing converter: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("artelingo_retrieval_converter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(painting, caption, image=None, art_style="Impressionism"):
    return {
        "image": image or f"{art_style}/{painting}.jpg",
        "caption": caption,
        "image_id": f"{painting}#1",
        "art_style": art_style,
        "painting": painting,
    }


class ArtELingoRetrievalConversionTest(unittest.TestCase):
    def test_keeps_first_five_ordered_captions_and_drops_short_paintings(self):
        converter = load_converter()
        source = []
        for n in range(6):
            source.append(row("six", f"six-{n}"))
            if n < 4:
                source.append(row("four", f"four-{n}"))
            if n < 5:
                source.append(row("five", f"five-{n}"))

        output, dropped = converter.build_retrieval_rows(source)

        self.assertEqual(dropped, 1)
        self.assertEqual(output, [
            {
                "image": "Impressionism/six.jpg",
                "caption": [f"six-{n}" for n in range(5)],
                "image_id": "six",
                "art_style": "Impressionism",
                "painting": "six",
            },
            {
                "image": "Impressionism/five.jpg",
                "caption": [f"five-{n}" for n in range(5)],
                "image_id": "five",
                "art_style": "Impressionism",
                "painting": "five",
            },
        ])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.json"
            path.write_text(json.dumps(output), encoding="utf-8")
            self.assertEqual(converter.verify_retrieval_file(path), 2)

    def test_rejects_conflicting_art_style_within_painting(self):
        converter = load_converter()
        source = [row("same", f"caption-{n}") for n in range(5)]
        source[3]["art_style"] = "Realism"

        with self.assertRaisesRegex(ValueError, "same"):
            converter.build_retrieval_rows(source)

    def test_rejects_shared_image_across_paintings(self):
        converter = load_converter()
        source = [row(painting, f"{painting}-{n}", image="shared.jpg")
                  for painting in ("first", "second") for n in range(5)]

        with self.assertRaisesRegex(ValueError, "shared.jpg"):
            converter.build_retrieval_rows(source)

    def test_file_verification_still_rejects_invalid_rows_under_optimized_python(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.json"
            path.write_text(json.dumps([
                {"image": "a.jpg", "image_id": "a", "caption": ["only one"]},
            ]), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-O", "-c",
                 "import runpy, sys; runpy.run_path(sys.argv[1])['verify_retrieval_file'](sys.argv[2])",
                 str(SCRIPT), str(path)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_invalid_test_split_does_not_replace_existing_val_output(self):
        converter = load_converter()
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "artelingo_val.json").write_text(
                json.dumps([row("valid", f"caption-{n}") for n in range(5)]),
                encoding="utf-8",
            )
            bad_test_rows = [row("invalid", f"caption-{n}") for n in range(5)]
            bad_test_rows[4]["art_style"] = "Realism"
            (data_dir / "artelingo_test.json").write_text(
                json.dumps(bad_test_rows), encoding="utf-8",
            )
            val_output = data_dir / "artelingo_val_retrieval.json"
            val_output.write_text("previous valid output", encoding="utf-8")
            converter.DATA_DIR = data_dir

            with self.assertRaisesRegex(ValueError, "invalid"):
                converter.main()
            self.assertEqual(val_output.read_text(encoding="utf-8"), "previous valid output")


if __name__ == "__main__":
    unittest.main()
