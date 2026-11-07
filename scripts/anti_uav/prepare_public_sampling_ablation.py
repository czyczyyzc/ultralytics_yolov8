#!/usr/bin/env python3
"""Prepare isolated sampling/color configurations and audit real-data quota coverage."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.audit_labeler_expansion_regression import AuditDataset
from scripts.anti_uav.label_pool_sampling import FixedDomainQuotaSampler
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def prepare(old_run, new_run, output):
    output.mkdir(parents=True, exist_ok=False)
    old_protocol=json.loads((old_run/'protocol.json').read_text())
    new_protocol=json.loads((new_run/'protocol.json').read_text())
    old=yaml.safe_load(Path(old_protocol['data']).read_text())
    source=yaml.safe_load(Path(new_protocol['data']).read_text())
    for config in (old,source):
        listing=Path(config['train']);pin=config['trusted_label_caches'][str(listing)]
        if sha256(listing)!=pin['source_sha256'] or sha256(listing.with_suffix('.cache'))!=pin['cache_sha256']:
            raise ValueError('Audited source list/cache changed')
    source_dir=Path(new_protocol['data']).parent
    public=source_dir/'public_accepted.json'
    quotas=dict(gray_tiny=19,gray_sub4=3,gray_other=16,antiuav=7,public=9,negative=10)
    source_summary=json.loads((source_dir/'summary.json').read_text())
    fixed=dict(source['tiny_sampling'],policy='fixed_domain_v1',batch_quotas=quotas,
        reference_epoch_slots=source_summary['epoch_slots'],public_manifest=str(public),
        public_manifest_sha256=sha256(public),native_positive_source_list=old['train'],
        native_positive_source_sha256=sha256(old['train']),max_extra_copies_per_native_index=1)
    configs={}
    for sampler in ('order','quota'):
        for color in ('source','gray3'):
            data=deepcopy(source)
            data['tiny_sampling']=dict(source['tiny_sampling'],policy='even_exposure_v1') if sampler=='order' else fixed
            data['input_color']=color
            name=sampler+'_'+color
            path=output/(name+'.yaml')
            path.write_text(yaml.safe_dump(data,sort_keys=False))
            configs[name]=dict(path=str(path),sha256=sha256(path))
    labels=np.load(Path(source['train']).with_suffix('.cache'),allow_pickle=True).item()['labels']
    sampler=FixedDomainQuotaSampler(AuditDataset(labels),source['label_sampling'],fixed,64,20261015)
    public_seen=set();epochs=[]
    native=[i for key in sampler.GROUPS[:4] for i in sampler.groups[key]]
    old_native_counts=Counter(labels[i]['im_file'] for i in native)
    for epoch in range(sampler.audit['public_full_coverage_epochs']):
        sampler.set_epoch(epoch)
        order=list(sampler)
        groups=[sampler.group_by_index[i] for i in order]
        for start in range(0,len(order),64):
            if Counter(groups[start:start+64])!=quotas:
                raise AssertionError('Actual batch violates fixed quota')
        counts=Counter(order)
        if any(not 1<=counts[i]<=2 for i in native):
            raise AssertionError('Native exposure floor/replay limit violated')
        replay=Counter(labels[i]['im_file'] for i in order if sampler.group_by_index[i] in sampler.GROUPS[:4])
        if any(replay[path]<count for path,count in old_native_counts.items()):
            raise AssertionError('Native repeated positive exposure lost')
        public_selected=[i for i in order if sampler.group_by_index[i]=='public']
        if len(public_selected)!=len(set(public_selected)):
            raise AssertionError('Public duplicate inside epoch')
        public_seen.update(public_selected)
        epochs.append(dict(epoch=epoch,slots=len(order),counts=dict(Counter(groups)),
            first_batch=dict(Counter(groups[:64])),last_batch=dict(Counter(groups[-64:])),
            public_covered_so_far=len(public_seen),all_native_positive_indices_present=True,
            native_extra_copies_max=max(counts[i]-1 for i in native),
            index_sha256=hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()))
        dump(output/'partial.json',dict(sampler_audit=sampler.audit,epochs=epochs))
    if public_seen!=set(sampler.groups['public']):
        raise AssertionError('Public coverage incomplete across rotation cycle')
    evaluation=yaml.safe_load((source_dir/'fixedwing_public_holdout.yaml').read_text())
    evaluation['input_color']='gray3'
    (output/'fixedwing_eval_gray3.yaml').write_text(yaml.safe_dump(evaluation,sort_keys=False))
    report=dict(complete=True,training_started=False,old_run=str(old_run),new_run=str(new_run),
        unchanged_source_train_sha256=sha256(source['train']),configs=configs,
        sampler_audit=sampler.audit,epochs=epochs,all_public_positives_covered=True,
        input_color='gray3 means BGR->luminance->three equal channels after augmentation, before Format; no source files changed',
        comparison_rules='Compare all candidates and controls with the SAME input color and frame list; no test-driven selection',
        changes_isolated=True)
    dump(output/'preparation.json',report)
    print(json.dumps(dict(complete=True,training_started=False,output=str(output),sampler=sampler.audit)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--new-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    prepare(a.old_run.resolve(),a.new_run.resolve(),a.output.resolve())


if __name__=='__main__':main()
