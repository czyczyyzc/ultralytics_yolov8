import json

import pytest

from scripts.anti_uav.continue_repaired_training import completed_foundation, delivery_command
from scripts.anti_uav.run_labeler_retrain_delivery import augmentation_policy


def foundation(tmp_path, epochs=30):
    weights = tmp_path/'foundation/p3/weights/best.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'checkpoint')
    (weights.parent.parent/'results.csv').write_text('epoch,loss\n'+''.join(f'{i},1\n' for i in range(1, epochs+1)))
    audit = tmp_path/'audit'
    audit.mkdir()
    (audit/'video00009_foundation_p3.json').write_text(json.dumps({'fitness': .6, 'native/c0.03/R': .6, 'native/mAP50': .7}))
    (tmp_path/'status.json').write_text(json.dumps(dict(stage='foundation_complete')))
    return tmp_path


def test_partial_best_checkpoint_cannot_skip_unfinished_foundation(tmp_path):
    run = foundation(tmp_path, 8)
    with pytest.raises(ValueError, match='all 30 epochs'):
        completed_foundation(run)


def test_complete_foundation_requires_metrics_and_weights(tmp_path):
    run = foundation(tmp_path)
    result = completed_foundation(run)
    assert result['completed_epochs'] == 30 and len(result['weights_sha256']) == 64
    (run/'foundation/p3/weights/best.pt').unlink()
    with pytest.raises(ValueError, match='missing'):
        completed_foundation(run)


def test_continuation_resumes_the_same_repaired_run_not_a_new_foundation():
    command = delivery_command('python', '/repaired', '/data', '/delivery', '/public/yolov8n.pt', '/old/eval.pt', '/toolkit/python', '0')
    assert '--resume' in command
    assert '--foundation-only' not in command
    assert command[command.index('--run-dir')+1] == '/repaired'
    assert command[command.index('--foundation')+1] == '/public/yolov8n.pt'
    assert command[command.index('--baseline-student')+1] == '/old/eval.pt'


def test_handover_describes_repaired_cache_without_claiming_all_frames_replaced():
    manifest = dict(sampler_audit=dict(policy='balanced_v2', tiny_fraction=.30, negative_fraction=.15, max_extra_repeat_per_image=1))
    text = augmentation_policy(manifest, dict(cache_ready=True, new_ready_backgrounds=2815, added_positive=24307))
    assert '2815帧有安全背景缓存' in text
    assert '21492帧保留原图和原框' in text
    assert '30.0%' in text and '15.0%' in text
