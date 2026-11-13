#!/usr/bin/env python3
"""Prepare frozen fresh-seed validation and legal-event history comparisons."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from align_system.interfaces.cage_observations import capture_paths, format_window
from legal_event_history import event_history, RAW_GUIDE
from prepare_blue_prompt_screen import INSTRUCTIONS, COMPACT, LABEL_BY_DIRECTORY, render, VARIANTS
from run_classifier_replay import LABELS, digest, read_requests, write_json


EVENT_VARIANTS={'T-short':(False,7),'R-short':(True,7),
                'T-prefix':(False,None),'R-prefix':(True,None)}


def identify(row):
    row['provenance']['messages_sha256']=hashlib.sha256(json.dumps(row['messages'],sort_keys=True).encode()).hexdigest()
    row['request_id']=hashlib.sha256(json.dumps({k:row[k] for k in
        ['variant','partition','seed','pair_id','messages','generation']},sort_keys=True).encode()).hexdigest()
    return row


def load_old(directory):
    summary=json.loads((directory/'summary.json').read_text())
    return {(e['class_directory'],e['trajectory_id']):json.loads(p.read_text())
            for e,p in zip(summary['episodes'],capture_paths(directory))}


def load_fresh(directory):
    summary=json.loads((directory/'summary.json').read_text())
    if summary.get('status')!='complete' or summary.get('condition')!='fresh_monitor':
        raise ValueError('Fresh collection must be complete')
    captures=[]
    for episode in summary['episodes']:
        p=directory/episode['capture_path']
        if digest(p)!=episode['capture_sha256']:raise ValueError('Fresh capture changed')
        capture=json.loads(p.read_text())
        if capture.get('repeat_capture_identical') is not True or capture['seed']!=episode['seed']:
            raise ValueError('Fresh repetition or seed verification failed')
        if capture['cage_python_source_sha256']!=summary['cage_python_source_sha256']:
            raise ValueError('Fresh simulator fingerprint differs')
        captures.append((episode,capture))
    if len(captures)!=summary['expected_episodes'] or len({(e['seed'],e['true_label']) for e,_ in captures})!=len(captures):
        raise ValueError('Incomplete or duplicated fresh captures')
    for seed in summary['seed_groups']:
        if {e['true_label'] for e,_ in captures if e['seed']==seed}!=set(LABELS):
            raise ValueError('Fresh classes are unbalanced')
    return captures


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    read_requests(path)


def raw_inputs(args):
    captures=load_old(args.capture_dir)
    protocol=json.loads((args.screen_inputs/'protocol.json').read_text())
    training_seed=protocol['shared_seed_partitions']['examples'][0]
    systems={}
    for raw in (False,True):
        text=INSTRUCTIONS.replace("You receive only Blue's selected actions and its resulting observation tables.",
            "You receive Blue's actions, tables, and any explicitly supplied raw Blue sensor evidence.")+'\n'+COMPACT+'\n'+RAW_GUIDE
        text+='\nExamples use seven steps; a query can use any longer observed prefix with the same fields.\n'
        for label in LABELS:
            matches=[c for (directory,_),c in captures.items()
                     if LABEL_BY_DIRECTORY[directory]==label and c['seed']==training_seed]
            if len(matches)!=1:raise ValueError('Ambiguous example episode')
            text+='\nTraining example:\n'+event_history(matches[0],30,37,raw)+'\nCorrect training label: '+label+'\n'
        systems[raw]=text
    args.output.mkdir(parents=True,exist_ok=False)
    for partition in ['development','confirmation']:
        bases=[r for r in read_requests(args.screen_inputs/f'{partition}-requests.jsonl') if r['variant']=='P2']
        rows=[]
        for base in bases:
            capture=captures[(base['provenance']['class_directory'],base['trajectory_id'])]
            end=base['provenance']['window_end']
            for variant,(raw,length) in EVENT_VARIANTS.items():
                start=max(0,end-length) if length else 0
                row=copy.deepcopy(base)
                row.pop('original_request_id',None)
                row.update(variant=variant,prompt_protocol='legal_event_history',
                           input_view='table_plus_raw_blue_events' if raw else 'table_events',
                           actual_history_length=end-start,requested_history_length=length)
                row['messages']=[{'role':'system','content':systems[raw]},
                    {'role':'user','content':'# OBSERVATIONS\n'+event_history(capture,start,end,raw)+
                     '\n\n# AVAILABLE LABELS\n'+'\n'.join(LABELS)}]
                row['provenance'].update(window_start=start,window_size=end-start,
                                          example_episode_seeds=[training_seed])
                rows.append(identify(row))
        rows.sort(key=lambda r:(r['provenance']['window_end'],r['seed'],r['variant'],LABELS.index(r['true_label'])))
        write_rows(args.output/f'{partition}-requests.jsonl',rows)
    shutil.copyfile(args.screen_inputs/'baseline-predictions.jsonl',args.output/'baseline-predictions.jsonl')
    write_json(args.output/'protocol.json',{
        'experiment':'legal_events_by_history','variants':EVENT_VARIANTS,
        'shared_seed_partitions':protocol['shared_seed_partitions'],
        'screen_protocol_sha256':digest(args.screen_inputs/'protocol.json'),
        'capture_audit_sha256':digest(args.capture_dir/'summary.json'),
        'history_factor':'Query history only; seven-step examples fixed within each representation.',
        'raw_factor':'Tables plus legally observed raw fields versus tables; matched example episode identities.',
        'primary_selection':'Highest development joint accuracy; shorter inputs break ties; confirm one.',
        'secondary_selection':'Best raw development condition versus table condition at the same history length, on fresh seeds.',
        'fresh_primary_times':[17,37,58,79,100],'fresh_reduced_times':[37,79,100],
        'confirmation_limit':'Previously included in inspected aggregate baseline; exploratory.'})


def fresh_inputs(args):
    selection=json.loads((args.primary_output/'selection.json').read_text())
    winner=selection['selected_variants'][0]
    templates=[r for r in read_requests(args.screen_inputs/'development-requests.jsonl') if r['variant']==winner]
    if len({r['messages'][0]['content'] for r in templates})!=1:
        raise ValueError('Primary selected prompt is not fixed')
    template=templates[0]
    legacy=read_requests(args.legacy_requests)[0]
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for episode,capture in load_fresh(args.capture_dir):
        for end in [17,37,58,79,100]:
            pair=hashlib.sha256(f"fresh:{episode['seed']}:{episode['true_label']}:{end}".encode()).hexdigest()
            for variant,base in [('P0',legacy),(winner,template)]:
                row=copy.deepcopy(base)
                row.pop('original_request_id',None)
                compact,_,length=VARIANTS[winner] if variant!= 'P0' else (False,0,7)
                start=max(0,end-length)
                text=render(capture,end,end-start,compact)
                tail=base['messages'][-1]['content'].split('\n\n# AVAILABLE LABELS\n',1)[1]
                row.update(variant=variant,partition='fresh',seed=episode['seed'],pair_id=pair,
                           true_label=episode['true_label'],trajectory_id=episode['trajectory_id'],
                           scenario_id=episode['trajectory_id'],historical_prediction=None,
                           prompt_protocol='fresh_seed_validation',condition='fresh_monitor',actual_history_length=end-start,
                           requested_history_length=length)
                row['messages']=[copy.deepcopy(base['messages'][0]),
                                 {'role':'user','content':'# OBSERVATIONS\n'+text+'\n\n# AVAILABLE LABELS\n'+tail}]
                row['provenance']={'window_start':start,'window_end':end,'window_size':end-start,
                                   'capture_sha256':episode['capture_sha256'],
                                   'class_directory':episode['class_directory']}
                rows.append(identify(row))
    rows.sort(key=lambda r:(r['provenance']['window_end'],r['seed'],r['variant'],LABELS.index(r['true_label'])))
    write_rows(args.output/'fresh-requests.jsonl',rows)
    write_json(args.output/'protocol.json',{'experiment':'fresh_seed_primary','primary_variant':winner,
        'primary_selection_sha256':digest(args.primary_output/'selection.json'),
        'fresh_capture_summary_sha256':digest(args.capture_dir/'summary.json'),
        'primary_prompt_sha256':hashlib.sha256(template['messages'][0]['content'].encode()).hexdigest(),
        'baseline_prompt_sha256':hashlib.sha256(legacy['messages'][0]['content'].encode()).hexdigest(),
        'candidate_selection':'Original development only; no confirmation or fresh scores used.',
        'times':[17,37,58,79,100],'reduced_times':[37,79,100],
        'seed_count':len({r['seed'] for r in rows}),'comparison':'Frozen original screen winner versus unchanged historical Blue prompt.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['raw','fresh'])
    p.add_argument('--screen-inputs',required=True,type=Path)
    p.add_argument('--capture-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--primary-output',type=Path)
    p.add_argument('--legacy-requests',type=Path)
    args=p.parse_args()
    if args.kind=='raw':raw_inputs(args)
    else:
        if args.primary_output is None or args.legacy_requests is None:p.error('Fresh preparation needs primary output and legacy requests')
        fresh_inputs(args)
    print(json.dumps({'kind':args.kind,'output':str(args.output)}),flush=True)


if __name__=='__main__':
    main()
