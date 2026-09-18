import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from scripts.anti_uav.online_gray_replacement import (
    OnlineReplacementDataset, compose_cached, decode_prepared, encode_prepared, select_variant,
)
from scripts.anti_uav.synthesize_gray_drone_replacements import (
    SkipSample, normalize_box, prepare_target, render_prepared, replace_target, sha256,
)
from scripts.anti_uav.label_pool_sampling import set_native_sampler_epoch


def scene():
    gray = np.clip(150+np.random.default_rng(12).normal(0,.7,(240,320)),0,255).astype(np.uint8)
    gray[112:128,148:172] = 100
    rgba = np.zeros((40,80,4),np.uint8)
    rgba[12:28,:] = [60,60,60,255]
    rgba[:,30:50] = [90,90,90,255]
    return gray, [148.,112.,24.,16.], rgba


def test_split_render_and_cache_are_pixel_identical():
    gray, box, rgba = scene()
    original = gray.copy()
    full, mask, bbox, metrics = replace_target(gray,box,rgba,123)
    prepared = prepare_target(gray,box,123)
    saved = decode_prepared(encode_prepared(prepared))
    bgr, normalized, online_metrics = compose_cached(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),saved,rgba)
    assert np.array_equal(bgr[:,:,0],full)
    assert np.array_equal(gray,original)
    assert metrics == online_metrics
    assert np.array_equal(full[mask==0],gray[mask==0])
    expected = np.array([float(v) for v in normalize_box(bbox,320,240).split()[1:]])
    np.testing.assert_allclose(normalized,expected,atol=1e-7)


def test_all_assets_cycle_without_epoch_expansion():
    picks = [select_variant(17,"source",e,0,53) for e in range(106)]
    assert picks.count(None) == 53
    assert sorted(p for p in picks if p is not None) == list(range(53))
    assert picks == [select_variant(17,"source",e,0,53) for e in range(106)]
    assert select_variant(1,"a",0,0,53,0) is None
    assert all(select_variant(1,"a",e,0,53,1) is not None for e in range(53))


def test_source_mismatch_fails_not_silent_synthesis():
    gray,box,rgba=scene()
    prepared=prepare_target(gray,box,2)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    image[120,150]=1
    with pytest.raises(ValueError,match="Source pixels"):
        compose_cached(image,prepared,rgba)


def test_size_gates_do_not_delete_originals():
    gray,_,_=scene()
    with pytest.raises(SkipSample,match="target_size"):
        prepare_target(gray,[1.,1.,200.,180.],1)


class Base:
    rect=False
    augment=True
    use_segments=False
    use_keypoints=False
    use_obb=False
    def __init__(self,labels):
        self.labels=labels
        self.im_files=[r["im_file"] for r in labels]
        self.collate_fn=lambda x:x
    def _imgsz_hw(self): return (544,960)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {"original_index":i}
    def transforms(self,x): return x
    def update_labels_info(self,x): return x


def cache_fixture(tmp_path,monkeypatch):
    gray,box,rgba=scene()
    image=tmp_path/"original.png"
    Image.fromarray(gray).save(image)
    prepared=prepare_target(gray,box,12)
    payload=encode_prepared(prepared)
    payload_sha=hashlib.sha256(payload).hexdigest()
    db=sqlite3.connect(tmp_path/"cache.sqlite")
    db.execute("CREATE TABLE backgrounds (frame INTEGER, state TEXT, payload BLOB, payload_sha TEXT)")
    db.execute("INSERT INTO backgrounds VALUES (1,'ready',?,?)",(payload,payload_sha))
    db.commit();db.close()
    catalog=tmp_path/"catalog.json"
    catalog.write_text("{}")
    monkeypatch.setattr("scripts.anti_uav.online_gray_replacement.load_assets",lambda *_:[({"id":"1"},rgba)])
    info=dict(pack="cache.sqlite",frame=1,video_sha256="train",box=box,
              source_sha256=sha256(image),payload_sha=payload_sha)
    index=dict(images={str(image):info},heldout_sha256=["test","val"],catalog=str(catalog),
               catalog_sha256=sha256(catalog),asset_ids=["1"])
    (tmp_path/"index.json").write_text(json.dumps(index))
    (tmp_path/"summary.json").write_text(json.dumps(dict(stage="complete",is_smoke_subset=False,
                                                         index_sha256=sha256(tmp_path/"index.json"))))
    bbox=[float(v) for v in normalize_box(box,320,240).split()[1:]]
    labels=[dict(im_file=str(image),shape=(240,320),bboxes=np.array([bbox]),cls=np.array([[0]]),
                 normalized=True,bbox_format="xywh"),
            dict(im_file="negative",shape=(240,320),bboxes=np.empty((0,4)),cls=np.empty((0,1)))]
    return Base(labels),dict(cache=str(tmp_path),replacement_probability=1),info


def test_wrapper_updates_box_preserves_original_labels_and_negatives(tmp_path,monkeypatch):
    base,config,_=cache_fixture(tmp_path,monkeypatch)
    old=base.labels[0]["bboxes"].copy()
    dataset=OnlineReplacementDataset(base,config)
    out=dataset[0]
    assert out["img"].shape[:2]==(544,726)
    assert out["bboxes"].shape==(1,4)
    assert not np.array_equal(out["bboxes"],old)
    assert np.array_equal(base.labels[0]["bboxes"],old)
    assert dataset[1]=={"original_index":1}
    assert len(dataset)==len(base)


def test_wrapper_rejects_changed_source(tmp_path,monkeypatch):
    base,config,_=cache_fixture(tmp_path,monkeypatch)
    dataset=OnlineReplacementDataset(base,config)
    Image.fromarray(np.zeros((240,320),np.uint8)).save(base.im_files[0])
    with pytest.raises(ValueError,match="Source image changed"):
        dataset[0]


def test_wrapper_failed_cutout_keeps_original(tmp_path,monkeypatch):
    base,config,_=cache_fixture(tmp_path,monkeypatch)
    dataset=OnlineReplacementDataset(base,config)
    def fail(*_): raise SkipSample("asset_disappeared_after_downsampling")
    monkeypatch.setattr("scripts.anti_uav.online_gray_replacement.compose_cached",fail)
    assert dataset[0]=={"original_index":0}


def test_holdout_cache_rejected(tmp_path,monkeypatch):
    base,config,info=cache_fixture(tmp_path,monkeypatch)
    p=tmp_path/"index.json"
    index=json.loads(p.read_text())
    index["heldout_sha256"].append(info["video_sha256"])
    p.write_text(json.dumps(index))
    (tmp_path/"summary.json").write_text(json.dumps(dict(stage="complete",is_smoke_subset=False,index_sha256=sha256(p))))
    with pytest.raises(ValueError,match="Held-out"):
        OnlineReplacementDataset(base,config)


def test_epoch_callback_updates_dataset_and_sampler():
    class Epoch:
        def set_epoch(self,n): self.epoch=n
    sampler,dataset=Epoch(),Epoch()
    set_native_sampler_epoch(SimpleNamespace(epoch=7,train_loader=SimpleNamespace(sampler=sampler,dataset=dataset)))
    assert sampler.epoch==dataset.epoch==7
