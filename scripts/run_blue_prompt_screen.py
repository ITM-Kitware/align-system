#!/usr/bin/env python3
"""Run a bounded Blue prompt screen and freeze candidates before confirmation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from blue_prompt_screen_report import build, predictions
from run_classifier_replay import LABELS, digest, read_requests, write_json


def write_requests(path, rows):
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Recorded request selection changed: {path}')
    else:
        path.write_text(text)
    read_requests(path)


def preflight(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    sizes, over_limit, maximum = {}, set(), {}
    source_hashes = {}
    for partition in ['development', 'confirmation']:
        p = args.inputs/f'{partition}-requests.jsonl'
        source_hashes[partition] = digest(p)
        for row in read_requests(p):
            tokens = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if not isinstance(tokens,list) or not tokens or any(type(t) is not int for t in tokens):
                raise ValueError('Expected flat chat token IDs')
            count = len(tokens) + row['generation']['max_tokens']
            sizes[row['request_id']] = len(tokens)
            maximum[row['variant']] = max(maximum.get(row['variant'],0),count)
            if count > args.context_limit:
                over_limit.add(row['variant'])
    eligible = sorted(set(maximum)-over_limit)
    if not {'P1','P2','P3'} <= set(eligible):
        raise ValueError('Required core variants do not fit the declared context')
    result = {'source_sha256':source_hashes, 'context_limit':args.context_limit,
              'max_tokens_with_generation_by_variant':maximum, 'excluded_over_context':sorted(over_limit),
              'eligible_variants':eligible, 'input_tokens_by_request':sizes,
              'model_snapshot':str(args.model.resolve()), 'enable_thinking':False,
              'no_truncation':True}
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'context-audit.json'
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError('Preflight identity changed')
    write_json(target,result)
    print(json.dumps({key:value for key,value in result.items() if key!='input_tokens_by_request'}),flush=True)
    return result


def infer(args, rows, name):
    path=args.output/f'{name}-requests.jsonl'
    write_requests(path, rows)
    remaining=args.stop_at-time.time()-120
    if remaining <= 0:
        raise RuntimeError('Absolute GPU deadline reached')
    command=[sys.executable,str(Path(__file__).with_name('run_qwen_classifier_replay.py')),
             '--requests',str(path),'--output-dir',str(args.output/name),'--model',str(args.model),
             '--max-model-len',str(args.context_limit)]
    subprocess.run(command,check=True,timeout=remaining)


def execute(args):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Run inside a Slurm GPU allocation')
    if (args.output/'budget-decision.json').exists():
        raise ValueError('A workflow decision already exists; preserve this attempt and use a new output directory')
    audit=json.loads((args.output/'context-audit.json').read_text())
    for partition,sha in audit['source_sha256'].items():
        if digest(args.inputs/f'{partition}-requests.jsonl') != sha:
            raise ValueError('Preflight input changed')
    if audit['context_limit'] != args.context_limit or audit['model_snapshot'] != str(args.model.resolve()):
        raise ValueError('Preflight model or context changed')
    sizes=audit['input_tokens_by_request']
    development=read_requests(args.inputs/'development-requests.jsonl')
    confirmation=read_requests(args.inputs/'confirmation-requests.jsonl')
    eligible=audit['eligible_variants']
    pilot_by_label={}
    for label in LABELS:
        candidates=[r for r in development if r['variant'] in eligible and r['true_label']==label]
        pilot_by_label[label]=sorted(candidates,key=lambda r:(-sizes[r['request_id']],r['request_id']))[:4]
    pilot=[pilot_by_label[label][index] for index in range(4) for label in LABELS]
    infer(args,pilot,'pilot')
    measured=predictions(args.output/'pilot/predictions.jsonl')
    mean=sum(r['inference_seconds'] for r in measured.values())/len(measured)
    available=args.stop_at-time.time()
    variants=list(eligible)
    confirm_count=2
    ends={37,79}

    def estimate():
        n=sum(r['variant'] in variants and r['provenance']['window_end'] in ends for r in development)
        return (n+240*confirm_count)*mean*1.25+900

    for optional in ['P4','P6','P5']:
        if estimate()>available and optional in variants:
            variants.remove(optional)
    if estimate()>available:
        confirm_count=1
    if estimate()>available:
        ends={37}
    if estimate()>available:
        raise RuntimeError('Insufficient time for a complete balanced screen and one confirmation')
    selected=[r for r in development if r['variant'] in variants and r['provenance']['window_end'] in ends]
    decision={'eligible_variants':eligible,'scheduled_variants':variants,'development_window_ends':sorted(ends),
              'confirmation_candidates':confirm_count,'pilot_requests':len(pilot),
              'pilot_mean_inference_seconds':mean,'seconds_available':available,
              'conservative_estimated_seconds':estimate(),'pilot_accuracy_used_for_selection':False,
              'stop_at_epoch':args.stop_at,'slurm_job_id':os.environ['SLURM_JOB_ID']}
    write_json(args.output/'budget-decision.json',decision)
    print(json.dumps({'budget':decision}),flush=True)
    try:
        infer(args,selected,'development')
        comparison=build(args.inputs,args.output)
        metrics=comparison['partitions']['development']
        complete=[v for v in variants if metrics[v]['status']=='complete']
        if len(complete)!=len(variants):
            raise ValueError('Selection requires completed balanced development conditions')
        ranked=sorted(complete,key=lambda v:(-metrics[v]['joint_accuracy'],metrics[v]['mean_input_tokens'],v))
        winners=ranked[:confirm_count]
        selection={'selected_variants':winners,'development_ranking':ranked,
                   'criterion':'development joint accuracy; mean input tokens then variant ID break ties',
                   'confirmation_predictions_consulted':False,
                   'development_requests_sha256':digest(args.output/'development-requests.jsonl'),
                   'development_predictions_sha256':digest(args.output/'development/predictions.jsonl')}
        write_json(args.output/'selection.json',selection)
        infer(args,[r for r in confirmation if r['variant'] in winners],'confirmation')
    finally:
        build(args.inputs,args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--context-limit',type=int,default=32768)
    p.add_argument('--stop-at',type=float)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.preflight_only:
        preflight(args)
    else:
        if args.stop_at is None:
            p.error('--stop-at is required for inference')
        execute(args)


if __name__=='__main__':
    main()
