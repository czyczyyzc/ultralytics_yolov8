#!/usr/bin/env python3
"""CPU smoke/visual QA on a completed online-background cache, without training."""
import argparse
from copy import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anti_uav.build_gray_replacement_batch import load_assets
from scripts.anti_uav.online_gray_replacement import OnlineReplacementDataset, compose_cached, decode_prepared
from scripts.anti_uav.synthesize_gray_drone_replacements import SkipSample, preview_card, preview_overview, render_prepared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/"previews").mkdir()
    cv2.setNumThreads(2)
    index=json.loads((a.cache/"index.json").read_text())
    assets=load_assets(Path(index["catalog"]),{"24","25"})
    rows=list(index["images"].items())[:6]
    times,failures,previews=[],[],[]
    for source,info in rows:
        db=sqlite3.connect(f"file:{a.cache/info['pack']}?mode=ro",uri=True)
        payload=db.execute("SELECT payload FROM backgrounds WHERE frame=?",(info["frame"],)).fetchone()[0]
        db.close()
        prepared=decode_prepared(payload)
        image=cv2.imread(source)
        original=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        successes=0
        for asset,rgba in assets:
            start=time.perf_counter()
            try:
                patch,mask,box,metrics=render_prepared(prepared,rgba)
            except SkipSample as error:
                failures.append(dict(frame=info["frame"],asset=asset["id"],reason=str(error)))
                continue
            times.append((time.perf_counter()-start)*1000)
            result,normalized,check_metrics=compose_cached(image,prepared,rgba)
            x,y,side=(prepared["meta"][k] for k in ("left","top","side"))
            fullmask=np.zeros(original.shape,np.uint8)
            fullmask[y:y+side,x:x+side]=mask
            if not np.array_equal(result[fullmask==0],image[fullmask==0]):
                raise AssertionError("Compositor altered pixels outside the foreground repair")
            if metrics != check_metrics:
                raise AssertionError("Online and direct render mismatch")
            if successes<2:
                rel=f"previews/{len(previews)+1:02d}_f{info['frame']}_a{asset['id']}.png"
                meta=dict(video=info["video_sha256"][:12],frame=info["frame"],asset_id=asset["id"],
                          asset_model=asset["model"],metrics=metrics)
                preview_card(original,result[:,:,0],info["box"],meta,Image.fromarray(rgba),box).save(a.output/rel)
                previews.append(rel)
            successes+=1
    preview_overview(a.output,previews)
    # Exercise the real YOLO transforms and multiworker collation, not just the compositor.
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.dataset import YOLODataset
    from torch.utils.data import DataLoader
    hyp=copy(DEFAULT_CFG)
    hyp.mosaic=hyp.mixup=hyp.copy_paste=0.
    hyp.degrees=hyp.translate=hyp.scale=hyp.shear=hyp.perspective=0.
    hyp.hsv_h=hyp.hsv_s=hyp.hsv_v=0.
    hyp.fliplr=hyp.flipud=0.
    listing=a.output/"smoke_train.txt"
    listing.write_text("\n".join(source for source,_ in rows)*1+"\n")
    base=YOLODataset(img_path=str(listing),imgsz=(544,960),batch_size=2,augment=True,hyp=hyp,
                     rect=False,cache=False,data={"names":{0:"drone"}},task="detect",stride=32)
    wrapped=OnlineReplacementDataset(base,dict(cache=str(a.cache),replacement_probability=1.,allow_smoke_cache=True))
    cold={}
    for name,dataset in (("original",base),("replacement",wrapped)):
        elapsed=[]
        for rep in range(8):
            for i in range(len(base)):
                base.ims[i]=None
                t=time.perf_counter()
                result=dataset[i]
                elapsed.append((time.perf_counter()-t)*1000)
                if tuple(result["img"].shape)!=(3,544,960) or len(result["bboxes"])!=1:
                    raise AssertionError("YOLO shape/bbox contract failed")
        cold[name]=dict(mean_ms=float(np.mean(elapsed)),p95_ms=float(np.percentile(elapsed,95)))
    checked_batches=0
    for batch in DataLoader(wrapped,batch_size=2,num_workers=2,collate_fn=base.collate_fn):
        if tuple(batch["img"].shape[1:])!=(3,544,960):
            raise AssertionError("Multiworker batch shape mismatch")
        checked_batches+=1
    summary=dict(source_frames=len(rows),asset_count=len(assets),attempted=len(rows)*len(assets),
        successful=len(times),failed=len(failures),failures=failures,
        cached_roi_render_ms=dict(mean=float(np.mean(times)),p50=float(np.median(times)),p95=float(np.percentile(times,95))),
        single_process_cold_yolo_load_transform_ms=cold,real_yolo_multiworker_batches_checked=checked_batches,
        outside_edit_mask_unchanged=True,original_files_unmodified=True,training_started=False,
        limitations="Small CPU smoke sample, not end-to-end GPU training throughput or an accuracy evaluation")
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
