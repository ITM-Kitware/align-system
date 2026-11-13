#!/usr/bin/env python3
"""Run bounded legal-event screens and frozen fresh-seed comparisons."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from blue_prompt_screen_report import predictions
from legal_event_history import event_history
from prepare_overnight_followups import identify, load_fresh, fresh_inputs, EVENT_VARIANTS
from report_overnight_followups import build
from run_blue_prompt_screen import infer
from run_classifier_replay import LABELS,digest,read_requests,write_json


def audit(paths,model,limit):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    sizes={};maximum={};hashes={}
    for path in paths:
        hashes[path.name]=digest(path)
        for row in read_requests(path):
            tokens=tokenizer.apply_chat_template(row['messages'],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if not isinstance(tokens,list) or not tokens or any(type(t) is not int for t in tokens):raise ValueError('Unexpected tokenizer result')
            total=len(tokens)+row['generation']['max_tokens']
            if total>limit:raise ValueError(f'Context overflow without truncation: {row["variant"]} {total}')
            sizes[row['request_id']]=len(tokens)
            maximum[row['variant']]=max(maximum.get(row['variant'],0),total)
    return {'source_sha256':hashes,'input_tokens_by_request':sizes,
            'max_tokens_with_generation_by_variant':maximum,'context_limit':limit,
            'model_snapshot':str(model.resolve()),'no_truncation':True}


def pilot_rows(rows,sizes):
    by_label={label:sorted([r for r in rows if r['true_label']==label],
        key=lambda r:(-sizes[r['request_id']],r['request_id']))[:4] for label in LABELS}
    if any(len(values)!=4 for values in by_label.values()):raise ValueError('Pilot needs four requests per class')
    return [by_label[label][i] for i in range(4) for label in LABELS]


def pilot_mean(args,rows,sizes,name):
    selected=pilot_rows(rows,sizes)
    infer(args,selected,name)
    completed=predictions(args.output/name/'predictions.jsonl')
    if len(completed)!=24:raise ValueError('Incomplete timing pilot')
    return sum(r['inference_seconds'] for r in completed.values())/len(completed)


def choose_fresh(rows,mean,seconds):
    seeds=sorted({r['seed'] for r in rows})
    variants={r['variant'] for r in rows}
    for times in [[17,37,58,79,100],[37,79,100]]:
        cost_per_seed=6*len(times)*len(variants)*mean*1.30
        n=min(len(seeds),100,int(max(0,seconds-900)/cost_per_seed))
        if n>=50:
            selected=set(seeds[:n])
            return [r for r in rows if r['seed'] in selected and r['provenance']['window_end'] in times],{
                'shared_seed_groups':n,'prediction_times':times,'variants':sorted(variants),
                'pilot_mean_inference_seconds':mean,'margin_multiplier':1.30,
                'estimated_seconds_with_margin_and_cleanup':n*cost_per_seed+900,
                'fresh_accuracy_used_for_budget':False}
    raise ValueError('Insufficient time for fifty independent seed groups and three prediction times')


def run_fresh(args):
    if not (args.inputs/'fresh-requests.jsonl').exists():
        fresh_inputs(argparse.Namespace(screen_inputs=args.screen_inputs,primary_output=args.primary_output,
            legacy_requests=args.legacy_requests,capture_dir=args.fresh_captures,output=args.inputs))
    rows=read_requests(args.inputs/'fresh-requests.jsonl')
    context=audit([args.inputs/'fresh-requests.jsonl'],args.model,args.context_limit)
    write_json(args.output/'context-audit.json',context)
    primary=json.loads((args.inputs/'protocol.json').read_text())['primary_variant']
    source=read_requests(args.screen_inputs/'development-requests.jsonl')
    legacy=read_requests(args.legacy_requests)
    legacy_ids={r['pair_id'] for r in source if r['variant']==primary}
    calibration=[r for r in source if r['variant']==primary]+[
        {**r,'variant':'P0'} for r in legacy if r['request_id'] in legacy_ids]
    temp=args.output/'calibration-requests.jsonl'
    temp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in calibration))
    timing=audit([temp],args.model,args.context_limit)
    mean=pilot_mean(args,calibration,timing['input_tokens_by_request'],'pilot')
    selected,budget=choose_fresh(rows,mean,args.stop_at-time.time())
    write_json(args.output/'fresh-budget.json',budget)
    infer(args,selected,'fresh')


def run_raw(args):
    development=read_requests(args.inputs/'development-requests.jsonl')
    confirmation=read_requests(args.inputs/'confirmation-requests.jsonl')
    context=audit([args.inputs/'development-requests.jsonl',args.inputs/'confirmation-requests.jsonl'],args.model,args.context_limit)
    write_json(args.output/'context-audit.json',context)
    mean=pilot_mean(args,development,context['input_tokens_by_request'],'pilot')
    estimate=(len(development)+240)*mean*1.30+900
    if estimate>args.stop_at-time.time():raise ValueError('Screen and confirmation cannot fit the allocation')
    write_json(args.output/'budget-decision.json',{'development_requests':len(development),
        'confirmation_requests':240,'pilot_mean_inference_seconds':mean,
        'estimated_seconds_with_margin_and_cleanup':estimate,'pilot_accuracy_used_for_selection':False})
    infer(args,development,'development')
    result=build(args.inputs,args.output)
    metrics=result['stages']['development']['variants']
    if any(m['status']!='complete' for m in metrics.values()):raise ValueError('Incomplete development screen')
    ranked=sorted(metrics,key=lambda v:(-metrics[v]['joint_accuracy'],metrics[v]['mean_input_tokens'],v))
    winner=ranked[0]
    raw_winner=next(v for v in ranked if v.startswith('R-'))
    control='T-'+raw_winner.split('-',1)[1]
    write_json(args.output/'selection.json',{'selected_variant':winner,'development_ranking':ranked,
        'fresh_raw_variant':raw_winner,'fresh_matched_table_control':control,
        'criterion':'Development joint accuracy then shorter mean input; raw champion selected from raw conditions only.',
        'confirmation_or_fresh_scores_consulted':False,
        'development_predictions_sha256':digest(args.output/'development/predictions.jsonl')})
    infer(args,[r for r in confirmation if r['variant']==winner],'confirmation')
    build(args.inputs,args.output)
    templates={v:next(r for r in development if r['variant']==v) for v in [raw_winner,control]}
    rows=[]
    for episode,capture in load_fresh(args.fresh_captures):
        for end in [17,37,58,79,100]:
            pair=hashlib.sha256(f"event-fresh:{episode['seed']}:{episode['true_label']}:{end}".encode()).hexdigest()
            for variant,template in templates.items():
                raw,length=EVENT_VARIANTS[variant];start=max(0,end-length) if length else 0
                row=copy.deepcopy(template)
                row.pop('original_request_id',None)
                row.update(partition='fresh',seed=episode['seed'],pair_id=pair,
                    true_label=episode['true_label'],trajectory_id=episode['trajectory_id'],
                    scenario_id=episode['trajectory_id'],historical_prediction=None,condition='fresh_monitor',
                    prompt_protocol='fresh_seed_validation',actual_history_length=end-start)
                row['messages']=[copy.deepcopy(template['messages'][0]),{'role':'user','content':
                    '# OBSERVATIONS\n'+event_history(capture,start,end,raw)+'\n\n# AVAILABLE LABELS\n'+'\n'.join(LABELS)}]
                row['provenance']={'window_start':start,'window_end':end,'window_size':end-start,
                    'capture_sha256':episode['capture_sha256'],'class_directory':episode['class_directory']}
                rows.append(identify(row))
    rows.sort(key=lambda r:(r['provenance']['window_end'],r['seed'],r['variant'],LABELS.index(r['true_label'])))
    candidate=args.output/'fresh-candidates.jsonl';candidate.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    fresh_context=audit([candidate],args.model,args.context_limit)
    write_json(args.output/'fresh-context-audit.json',fresh_context)
    calibration=[r for r in development if r['variant'] in templates]
    fresh_mean=pilot_mean(args,calibration,context['input_tokens_by_request'],'fresh-pilot')
    selected,budget=choose_fresh(rows,fresh_mean,args.stop_at-time.time())
    write_json(args.output/'fresh-budget.json',budget)
    infer(args,selected,'fresh')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['raw','fresh'])
    for name in ['inputs','output','model','fresh-captures','screen-inputs','primary-output','legacy-requests']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--context-limit',type=int,default=32768)
    p.add_argument('--stop-at',type=float,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Use a Slurm GPU allocation')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fresh-budget.json').exists() or (args.output/'selection.json').exists():
        raise ValueError('Preserve completed decisions; use a new attempt rather than reselecting')
    args.stop_at=min(args.stop_at,time.time()+int(os.environ['REPLAY_MAX_JOB_SECONDS'])-120)
    write_json(args.output/'runtime-cutoff.json',{'effective_stop_at_epoch':args.stop_at,
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'source_runner_sha256':digest(Path(__file__))})
    try:
        if args.kind=='fresh':run_fresh(args)
        else:run_raw(args)
    finally:
        if (args.inputs/'protocol.json').exists():build(args.inputs,args.output)


if __name__=='__main__':main()
