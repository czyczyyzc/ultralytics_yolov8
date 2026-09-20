#!/usr/bin/env python3
"""Exercise ordered ready-worker dispatch and failure cleanup without an NPU."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import run_rknn_dist_optimized as runtime


class Capture:
    def __init__(self):
        self.index, self.closed = 0, False

    def isOpened(self):
        return True

    def get(self, key):
        return {runtime.cv2.CAP_PROP_FPS:100, runtime.cv2.CAP_PROP_FRAME_COUNT:30,
                runtime.cv2.CAP_PROP_FRAME_WIDTH:48, runtime.cv2.CAP_PROP_FRAME_HEIGHT:32}[key]

    def read(self):
        image = np.full((32,48,3), self.index, np.uint8)
        self.index += 1
        return True, image

    def release(self):
        self.closed = True


class Detector:
    def __init__(self, delay, fail):
        self.delay, self.fail, self.closed = delay, fail, False
        self.lock = threading.Lock()

    def infer(self, image):
        if not self.lock.acquire(blocking=False):
            raise AssertionError("Concurrent reuse of one context")
        try:
            index = int(image[0,0,0])
            time.sleep(self.delay)
            if index == self.fail:
                raise RuntimeError("Injected NPU failure")
            return np.array([[index,0,index+1,1,.5]],np.float32), np.zeros(3), 0
        finally:
            self.lock.release()

    def close(self):
        self.closed = True


class TestPipeline(unittest.TestCase):
    def exercise(self, fail=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.write_bytes(b"test model")
            output, cap = root / "result", Capture()
            detectors = [Detector(delay,fail) for delay in (.01,.001,.001)]
            argv = ["runner", "--model", str(model), "--library", str(model),
                    "--upstream", directory, "--video", str(model), "--output", str(output),
                    "--frames", "30", "--warmup", "0", "--inflight", "6",
                    "--dispatch", "ready", "--detector-only", "--save-observations"]
            with patch("sys.argv",argv), patch.object(runtime.cv2,"VideoCapture",return_value=cap), \
                 patch.object(runtime,"make_detectors",return_value=detectors), \
                 patch.object(runtime,"hardware",return_value={}), \
                 patch.object(runtime,"source_hashes",return_value={}), \
                 patch.object(runtime.os,"sched_setaffinity"), patch("builtins.print"):
                if fail is None:
                    runtime.main()
                else:
                    with self.assertRaisesRegex(RuntimeError,"Injected NPU failure"):
                        runtime.main()
            self.assertTrue(cap.closed)
            self.assertTrue(all(d.closed for d in detectors))
            if fail is None:
                rows = [json.loads(line) for line in (output/"observations.jsonl").read_text().splitlines()]
                self.assertEqual([r["frame_index"] for r in rows],list(range(30)))
                self.assertEqual([r["boxes_xyxy_score"][0][0] for r in rows],list(range(30)))
                summary = json.loads((output/"summary.json").read_text())
                counts = summary["npu_worker_frame_counts"]
                self.assertEqual(sum(counts),30)
                self.assertLess(counts[0],max(counts[1:]))

    def test_ready_dispatch_preserves_order(self):
        self.exercise()

    def test_failure_closes_workers_and_decoder(self):
        self.exercise(fail=7)


if __name__ == "__main__":
    unittest.main()
