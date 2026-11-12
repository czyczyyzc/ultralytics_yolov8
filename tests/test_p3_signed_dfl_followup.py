from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.anti_uav.frozen_p3_addon_p2_trainer import add_on_parameter_ids
from scripts.anti_uav.p3_signed_dfl import P3SignedDFLLoss
from scripts.anti_uav.p3_student_distillation import P3StudentDistillationLoss, transfer_student_to_p3
from scripts.anti_uav.regression_repair_trainers import repaired_loss
from scripts.anti_uav.run_p3_signed_dfl_followup import BUDGETS, SIZES, check_improvement
from scripts.anti_uav.train_frozen_p3_addon_p2 import transfer_frozen_p3_weights, verify_legacy_outputs
from ultralytics.nn.tasks import DetectionModel, yaml_model_load


ROOT = Path(__file__).resolve().parents[1]


def signed_model(config):
    recipe = yaml_model_load(ROOT/"ultralytics/cfg/models/v8"/config)
    recipe.update(scale="n", p3_signed_dfl_v1=True)
    model = DetectionModel(recipe, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    return model


def test_signed_teacher_keeps_p3_and_student_distills_the_full_distribution(tmp_path):
    foundation = signed_model("yolov8.yaml")
    teacher = signed_model("yolov8-frozen-p3-addon-p2.yaml")
    with torch.no_grad():
        foundation.model[-1].p3_negative_bin.bias.fill_(.25)
    transfer_frozen_p3_weights(foundation, teacher)
    assert verify_legacy_outputs(foundation, teacher, size=64)["bit_exact"]
    assert not any(id(p) in add_on_parameter_ids(teacher) for p in teacher.model[-1].p3_negative_bin.parameters())
    student = deepcopy(teacher).train()
    student.model[-1].auxiliary_training_only = True
    teacher.requires_grad_(False).eval()
    criterion = P3StudentDistillationLoss(student, teacher)
    manifest = tmp_path/"public.json"
    manifest.write_text("[]")
    source = dict(input_color="gray3", tiny_sampling=dict(public_manifest=str(manifest),
        public_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    criterion.primary = repaired_loss(student, source, criterion.primary, {}, loss_type=P3SignedDFLLoss)
    criterion.primary.begin_epoch(3)
    images = torch.rand(2,3,64,64)
    batch = dict(img=images, im_file=["/gray/a.jpg","/gray/b.jpg"], batch_idx=torch.tensor([0.,1.]),
        cls=torch.zeros(2,1), bboxes=torch.tensor([[6.5/64,6/64,3/64,2/64],[.5,.5,7/64,4/64]]))
    loss, items = criterion(student(images), batch)
    loss.backward()
    assert items.isfinite().all() and student.model[-1].p3_negative_bin.bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    native = signed_model("yolov8.yaml").eval()
    student.eval()
    transfer_student_to_p3(student, native)
    with torch.no_grad():
        decoded, raw = student(images)
        native_decoded, native_raw = native(images)
    assert len(raw) == len(native_raw) == 3
    torch.testing.assert_close(decoded, native_decoded, rtol=0, atol=0)
    for a,b in zip(raw,native_raw):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def metrics():
    baseline = {budget: dict(frames=1000,budget_fp=int(budget),fp=int(budget),fp_per_1000=float(budget),
        sizes={size:dict(gt=100,tp50=50,tp75=30,median_matched_center_error_input_px=.5) for size in SIZES})
        for budget in BUDGETS}
    candidate = deepcopy(baseline)
    for row in candidate.values():
        row["sizes"][SIZES[0]]["tp50"] += 7
    return dict(arms={name:dict(splits=dict(Video00009=dict(matched_fp=value)))
                      for name,value in (("control",baseline),("p3_signed_dfl",candidate))})


def test_gate_requires_material_recall_gain_without_size_or_localization_regression():
    results = metrics()
    assert check_improvement(results)["passed"]
    row = results["arms"]["p3_signed_dfl"]["splits"]["Video00009"]["matched_fp"]["200"]
    row["sizes"][SIZES[0]]["tp75"] -= 3
    assert not check_improvement(results)["passed"]
    row["sizes"][SIZES[0]]["tp75"] += 3
    row["fp_per_1000"] -= 10
    assert not check_improvement(results)["passed"]
