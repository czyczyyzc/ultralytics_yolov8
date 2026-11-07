from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from scripts.anti_uav.label_pool_sampling import FixedDomainQuotaSampler, make_exposure_sampler


class Dataset:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)


def fixture(tmp_path):
    labels, records = [], []
    def add(path, edge):
        boxes = [] if edge is None else [[.5, .5, edge/960, edge/544]]
        labels.append(dict(im_file=path, shape=(544,960), cls=np.zeros((len(boxes),1)),
            bboxes=np.asarray(boxes).reshape(-1,4), normalized=True, bbox_format="xywh"))
    for name, count, edge in [("tiny",20,6),("sub4",8,3),("other",16,90)]:
        for i in range(count):add(f"/gray/{name}/{i}.jpg",edge)
    for i in range(10):add(f"/anti_uav300_yolo/images/rgb/scene_rgb_{i}.jpg",50)
    native = tmp_path/"native.txt"
    native.write_text("\n".join(r['im_file'] for r in labels)+"\n")
    for i in range(100):
        path=f"/public/{i}.jpg";add(path,30)
        records.append(dict(image=path,dataset="test",group=str(i//10),split="train",boxes=1))
    for i in range(3):add(f"/negative/fixed/{i}.jpg",None)
    for i in range(100):add(f"/negative/pool/{i}.jpg",None)
    pool=tmp_path/"neg.txt";pool.write_text("\n".join(r['im_file'] for r in labels[-100:])+"\n")
    manifest=tmp_path/"public.json";manifest.write_text(json.dumps(records))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    label_config=dict(negative_pool=str(pool),negative_pool_count=100,negatives_per_epoch=20,anchor_slots=157)
    tiny_config=dict(policy="fixed_domain_v1",public_manifest=str(manifest),public_manifest_sha256=digest(manifest),
        native_positive_source_list=str(native),native_positive_source_sha256=digest(native),
        reference_epoch_slots=128,batch_quotas=dict(gray_tiny=4,gray_sub4=1,gray_other=3,antiuav=2,public=3,negative=3))
    return Dataset(labels), label_config, tiny_config


def test_exact_batch_quotas_and_cyclic_native_coverage(tmp_path):
    data,label_config,config=fixture(tmp_path)
    sampler=make_exposure_sampler(data,label_config,config,16,41)
    assert type(sampler) is FixedDomainQuotaSampler
    order=list(sampler)
    assert len(order)==128
    for start in range(0,len(order),16):
        assert Counter(sampler.group_by_index[i] for i in order[start:start+16])==config['batch_quotas']
    counts=Counter(order)
    assert all(1 <= counts[i] <= 2 for i in range(54))
    assert all(counts[i]==1 for i in range(154,157))
    assert len({i for i in order if sampler.group_by_index[i]=='negative'})==24
    assert order==list(sampler)
    sampler.set_epoch(1)
    assert order!=list(sampler)


def test_public_rotation_covers_every_image_without_per_epoch_duplicates(tmp_path):
    data,label_config,config=fixture(tmp_path)
    sampler=FixedDomainQuotaSampler(data,label_config,config,16,41)
    seen=set()
    for epoch in range(sampler.audit['public_full_coverage_epochs']):
        sampler.set_epoch(epoch)
        public=[i for i in sampler if sampler.group_by_index[i]=='public']
        assert len(public)==len(set(public))==24
        seen.update(public)
    assert seen==set(sampler.groups['public'])


def test_fixed_quotas_reject_changed_manifest_and_unknown_positives(tmp_path):
    data,label_config,config=fixture(tmp_path)
    path=__import__('pathlib').Path(config['public_manifest'])
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='manifest changed'):
        FixedDomainQuotaSampler(data,label_config,config,16)
    config['public_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    data.labels[0]['im_file']='/unreviewed/new.jpg'
    with pytest.raises(ValueError,match='frozen native/public'):
        FixedDomainQuotaSampler(data,label_config,config,16)


def test_quota_and_replay_limits_fail_closed(tmp_path):
    data,label_config,config=fixture(tmp_path)
    with pytest.raises(ValueError,match='actual batch size'):
        FixedDomainQuotaSampler(data,label_config,config,64)
    config['reference_epoch_slots']=10000
    with pytest.raises(ValueError,match='bounded replay'):
        FixedDomainQuotaSampler(data,label_config,config,16)
