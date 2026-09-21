import json
from pathlib import Path
import tempfile
import unittest

from read_training_metrics import read_points


class MetricReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def write(self, text, stage="p3"):
        relative = "training_p3/p3" if stage == "p3" else "training_addon/p2"
        path = self.root / relative / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_before_first_epoch(self):
        self.assertEqual(read_points(self.root, 15), dict(items=[], latest_sequence=0))

    def test_spaces_and_nonfinite(self):
        self.write(" epoch, train/box_loss, unknown\n1,1.5,nan\n2,inf,-inf\n")
        result = read_points(self.root, 15)
        self.assertEqual(result["items"][0]["values"]["train/box_loss"], 1.5)
        self.assertIsNone(result["items"][1]["values"]["train/box_loss"])
        self.assertIsNone(result["items"][0]["values"]["unknown"])
        json.dumps(result, allow_nan=False)

    def test_incomplete_and_malformed_rows(self):
        self.write("epoch,train/box_loss\n1,2\n2\n3,no\n4,5")
        self.assertEqual(len(read_points(self.root, 15)["items"]), 1)

    def test_two_stages_and_cursor(self):
        self.write("epoch,train/box_loss\n15,1\n")
        self.write("epoch,train/box_loss\n1,2\n", "addon_p2")
        result = read_points(self.root, 15, after_sequence=15)
        self.assertEqual(result["latest_sequence"], 16)
        self.assertEqual(result["items"][0]["stage"], "addon_p2")
        self.assertEqual(result["items"][0]["epoch"], 1)
        self.assertEqual(read_points(self.root, 15, 16)["items"], [])

    def test_duplicate_and_invalid_epochs(self):
        self.write("epoch,x\n1,2\n1,3\n0,5\nnan,6\n16,7\n1.5,8\n")
        result = read_points(self.root, 15)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["values"]["x"], 3)

    def test_reconstructed_fitness(self):
        self.write("epoch,native/c0.03/F2,native/mAP50,native/mAP50-95\n1,0.6,0.5,0.2\n")
        values = read_points(self.root, 15)["items"][0]["values"]
        self.assertAlmostEqual(values["selection/fitness_reconstructed"], .49)


if __name__ == "__main__":
    unittest.main()
