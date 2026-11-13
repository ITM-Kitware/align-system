#!/usr/bin/env python3
"""Run a pilot-gated frozen classifier comparison on active-defender captures."""

import argparse
import json
import os
from pathlib import Path
import time

from active_defender_transfer import ARMS,report
from blue_prompt_screen_report import predictions
from run_blue_prompt_screen import infer,write_requests
from run_classifier_replay import LABELS,digest,read_requests,write_json
from run_overnight_followups import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','output','model']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stop-at',type=float)
    p.add_argument('--context-limit',type=int,default=32768)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.inputs/'requests.jsonl'
    if args.preflight_only:
        context=audit([source],args.model,args.context_limit)
        write_json(args.output/'context-audit.json',context)
        print(json.dumps({k:v for k,v in context.items() if k!='input_tokens_by_request'}));return
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Run inside a Slurm GPU allocation')
    if args.stop_at is None:p.error('--stop-at is required')
    if (args.output/'budget.json').exists():raise ValueError('Preserve this attempt; use a new output directory')
    args.stop_at=min(args.stop_at,time.time()+int(os.environ['TRANSFER_MAX_JOB_SECONDS'])-120)
    context=json.loads((args.output/'context-audit.json').read_text())
    if context['source_sha256'][source.name]!=digest(source):raise ValueError('Preflight requests changed')
    if context['model_snapshot']!=str(args.model.resolve()) or context['context_limit']!=args.context_limit:
        raise ValueError('Model or context differs from preflight')
    rows=read_requests(source);sizes=context['input_tokens_by_request'];selected={}
    for arm in ARMS:
        for label in LABELS:
            selected[arm,label]=sorted([r for r in rows if r['variant']==arm and r['true_label']==label],
                key=lambda r:(-sizes[r['request_id']],r['request_id']))[:2]
    pilot=[selected[arm,label][i] for i in range(2) for arm in ARMS for label in LABELS]
    infer(args,pilot,'pilot')
    done=predictions(args.output/'pilot/predictions.jsonl')
    if len(done)!=36:raise ValueError('Incomplete timing pilot')
    means={arm:sum(done[r['request_id']]['inference_seconds'] for r in pilot if r['variant']==arm)/12 for arm in ARMS}
    available=args.stop_at-time.time()
    def estimate(requests):
        return sum(means[r['variant']] for r in requests)*1.30+900
    mode='all_non_example'
    if estimate(rows)>available:
        rows=[r for r in rows if r['cohort']=='reserved'];mode='reserved_only'
    if estimate(rows)>available:raise ValueError('Complete reserved comparison cannot fit the remaining allocation')
    write_json(args.output/'budget.json',{'admitted_cohort':mode,'requests':len(rows),'pilot_requests':36,
        'pilot_mean_seconds_by_arm':means,'conservative_estimated_seconds':estimate(rows),
        'available_seconds':available,'pilot_accuracy_used_for_admission':False,
        'effective_stop_at_epoch':args.stop_at,'runner_sha256':digest(Path(__file__))})
    write_requests(args.output/'selected-requests.jsonl',rows)
    try:infer(args,rows,'inference')
    finally:report(args)


if __name__=='__main__':main()
