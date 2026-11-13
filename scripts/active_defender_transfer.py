#!/usr/bin/env python3
"""Prepare and report frozen-classifier transfer to active hybrid PPO defense."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from align_system.interfaces.cage_observations import capture_paths
from blue_prompt_screen_report import predictions,score,grouped_difference
from prepare_blue_prompt_screen import render,LABEL_BY_DIRECTORY
from run_classifier_replay import LABELS,digest,read_requests,write_json


ARMS=('monitor_P4','active_P4','active_P0')


def load_captures(directory):
    summary=json.loads((directory/'summary.json').read_text())
    return summary,{(e['class_directory'],e['trajectory_id']):(e,json.loads(p.read_text()))
                    for e,p in zip(summary['episodes'],capture_paths(directory))}


def prepare(args):
    reference=read_requests(args.requests)
    selection=json.loads((args.screen_output/'selection.json').read_text())
    winner=selection['selected_variants'][0]
    if winner!='P4':raise ValueError('This protocol is frozen to the declared three-example P4 winner')
    templates=[r for r in read_requests(args.screen_inputs/'development-requests.jsonl') if r['variant']==winner]
    systems={r['messages'][0]['content'] for r in templates}
    legacy_systems={r['messages'][0]['content'] for r in reference}
    if len(systems)!=1 or len(legacy_systems)!=1:raise ValueError('Frozen system prompts differ across source requests')
    protocol=json.loads((args.screen_inputs/'protocol.json').read_text())
    excluded=set(protocol['shared_seed_partitions']['examples'])
    reserved=set(protocol['shared_seed_partitions']['confirmation'])
    development=set(protocol['shared_seed_partitions']['development'])
    monitor_summary,monitors=load_captures(args.monitor_captures)
    active_summary,actives=load_captures(args.active_captures)
    if active_summary.get('condition')!='active_hybrid_ppo':raise ValueError('Expected fixed active hybrid PPO captures')
    if active_summary['monitor_audit_sha256']!=digest(args.monitor_captures/'summary.json'):
        raise ValueError('Active captures use a different Monitor audit')
    rows=[];episode_groups={};checkpoints={}
    for base in reference:
        key=(base['provenance']['class_directory'],base['trajectory_id'])
        mon_meta,monitor=monitors[key];active_meta,active=actives[key]
        seed=monitor['seed']
        if seed in excluded:continue
        if seed not in reserved|development:raise ValueError('Unexpected evaluation seed')
        if LABEL_BY_DIRECTORY[key[0]]!=base['true_label']:raise ValueError('Attacker class metadata differs')
        if active['seed']!=seed or active['monitor_capture_sha256']!=mon_meta['capture_sha256']:
            raise ValueError('Active and Monitor captures are not paired')
        for field in ['scenario_sha256','cage_python_source_sha256','cage_revision']:
            if active[field]!=monitor[field]:raise ValueError('Paired source or scenario differs')
        checkpoint=active['training_seed']
        if checkpoint in checkpoints and checkpoints[checkpoint]!=active['checkpoint_sha256']:
            raise ValueError('One training seed refers to different defender checkpoints')
        checkpoints[checkpoint]=active['checkpoint_sha256']
        if active_meta['training_seed']!=checkpoint:raise ValueError('Checkpoint metadata differs')
        episode_groups[(seed,base['true_label'])]={'seed':seed,'training_seed':checkpoint,
            'cohort':'reserved' if seed in reserved else 'development_exposed'}
        end=base['provenance']['window_end'];start=end-7
        pair=hashlib.sha256(f'active-transfer:{seed}:{base["true_label"]}:{end}'.encode()).hexdigest()
        for arm in ARMS:
            captured=monitor if arm=='monitor_P4' else active
            compact=arm!='active_P0'
            text=render(captured,end,7,compact)
            row=copy.deepcopy(base);row.pop('original_request_id',None)
            row.update(variant=arm,partition='active_transfer',condition='monitor' if arm=='monitor_P4' else 'active_hybrid_ppo',
                prompt_protocol='frozen_active_transfer',input_view='defender_table_compact_history' if compact else 'defender_table_and_blue_action',
                seed=seed,pair_id=pair,training_seed=checkpoint,cohort=episode_groups[(seed,base['true_label'])]['cohort'],
                historical_prediction=None,actual_history_length=7,requested_history_length=7,
                scenario_id=f'active-transfer/{arm}/{seed}/{base["true_label"]}/{end}')
            row['generation']['max_tokens']=256
            tail=(templates[0] if compact else base)['messages'][-1]['content'].split('\n\n# AVAILABLE LABELS\n',1)[1]
            row['messages']=[{'role':'system','content':next(iter(systems)) if compact else next(iter(legacy_systems))},
                {'role':'user','content':'# OBSERVATIONS\n'+text+'\n\n# AVAILABLE LABELS\n'+tail}]
            row['provenance']={'class_directory':key[0],'window_start':start,'window_end':end,'window_size':7,
                'capture_sha256':mon_meta['capture_sha256'] if arm=='monitor_P4' else active_meta['capture_sha256'],
                'checkpoint_sha256':active['checkpoint_sha256'],'source_request_id':base['request_id'],
                'messages_sha256':hashlib.sha256(json.dumps(row['messages'],sort_keys=True).encode()).hexdigest()}
            row['request_id']=hashlib.sha256(json.dumps({k:row[k] for k in
                ['variant','seed','pair_id','messages','generation']},sort_keys=True).encode()).hexdigest()
            rows.append(row)
    counts=Counter(r['variant'] for r in rows)
    if counts!=Counter({arm:480 for arm in ARMS}) or len(episode_groups)!=96:
        raise ValueError('Expected 96 non-example episodes and 480 requests per arm')
    rows.sort(key=lambda r:(r['provenance']['window_end'],r['seed'],r['variant'],LABELS.index(r['true_label'])))
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'requests.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));read_requests(path)
    coverage={name:dict(Counter(e['training_seed'] for e in episode_groups.values() if name=='all_non_example' or e['cohort']=='reserved'))
              for name in ['all_non_example','reserved']}
    write_json(args.output/'protocol.json',{'experiment':'frozen_monitor_classifier_active_transfer','arms':ARMS,
        'primary_cohort':'reserved','broader_cohort':'all_non_example','episodes':96,'requests_per_arm':480,
        'reserved_episodes':sum(e['cohort']=='reserved' for e in episode_groups.values()),
        'checkpoint_episode_coverage':coverage,'checkpoint_sha256':checkpoints,
        'excluded_example_seeds':sorted(excluded),'reserved_seeds':sorted(reserved),
        'development_exposed_seeds':sorted(development),'prediction_times':[17,37,58,79,100],
        'primary_selection_sha256':digest(args.screen_output/'selection.json'),
        'screen_protocol_sha256':digest(args.screen_inputs/'protocol.json'),
        'monitor_capture_summary_sha256':digest(args.monitor_captures/'summary.json'),
        'active_capture_summary_sha256':digest(args.active_captures/'summary.json'),
        'requests_sha256':digest(path),'preparation_script_sha256':digest(Path(__file__)),
        'system_prompt_sha256':{'P4':hashlib.sha256(next(iter(systems)).encode()).hexdigest(),
                               'P0':hashlib.sha256(next(iter(legacy_systems)).encode()).hexdigest()},
        'max_output_tokens':256,'examples_frozen_from_monitor':True,
        'interpretation':'Classification on recorded fixed-defender episodes, not live LLM-driven policy switching.',
        'coverage_limit':f'Reserved checkpoints: {sorted(coverage["reserved"])}. Broader checkpoints: {sorted(coverage["all_non_example"])}; broader cohort includes Monitor development seeds.'})
    print(json.dumps({'requests':len(rows),'coverage':coverage,'output':str(args.output)}),flush=True)


def report(args):
    all_rows=read_requests(args.inputs/'requests.jsonl')
    selected_path=args.output/'selected-requests.jsonl'
    rows=read_requests(selected_path) if selected_path.exists() else all_rows
    done=predictions(args.output/'inference/predictions.jsonl')
    if set(done)-{r['request_id'] for r in rows}:raise ValueError('Foreign predictions')
    result={'protocol':json.loads((args.inputs/'protocol.json').read_text()),'cohorts':{}}
    cohorts={'reserved':[r for r in all_rows if r['cohort']=='reserved'],'all_non_example':all_rows}
    for name,requests in cohorts.items():
        if not requests:continue
        metrics={arm:score([r for r in requests if r['variant']==arm],done) for arm in ARMS}
        contrasts={}
        for candidate,reference in [('active_P4','monitor_P4'),('active_P4','active_P0')]:
            baseline={r['pair_id']:done[r['request_id']] for r in requests if r['variant']==reference and r['request_id'] in done}
            contrast=grouped_difference([r for r in requests if r['variant']==candidate],done,baseline)
            if contrast is not None:contrast['interpretation']='Fixed-prompt paired transfer; shared-seed bootstrap. Cohort exposure is recorded separately.'
            contrasts[candidate+' minus '+reference]=contrast
        result['cohorts'][name]={'metrics':metrics,'joint_accuracy_contrasts':contrasts,
            'seed_groups':len({r['seed'] for r in requests}),
            'by_checkpoint':{str(c):{arm:score([r for r in requests if r['variant']==arm and r['training_seed']==c],done)
                                      for arm in ARMS} for c in sorted({r['training_seed'] for r in requests})},
            'by_prediction_time':{str(end):{arm:score([r for r in requests if r['variant']==arm and r['provenance']['window_end']==end],done)
                                      for arm in ARMS} for end in sorted({r['provenance']['window_end'] for r in requests})}}
    for name in ['budget.json','context-audit.json']:
        p=args.output/name
        if p.exists():
            value=json.loads(p.read_text());value.pop('input_tokens_by_request',None);result[name.removesuffix('.json')]=value
    write_json(args.output/'comparison.json',result)
    lines=['# Frozen classifier transfer to active defense','','Primary reserved cohort and broader development-exposed cohort are reported separately.','',
        '| Cohort | Arm | Status | Requests | Joint | Family | Objective |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for cohort,data in result['cohorts'].items():
        for arm,m in data['metrics'].items():
            values=[f"{100*m[k]:.2f}%" if m[k] is not None else '' for k in ['joint_accuracy','family_accuracy','objective_accuracy']]
            lines.append(f"| {cohort} | {arm} | {m['status']} | {m['n']}/{m['expected']} | "+' | '.join(values)+' |')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(args.output/'REPORT.md')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for name in ['requests','screen-inputs','screen-output','monitor-captures','active-captures','output']:
        prep.add_argument('--'+name,type=Path,required=True)
    rep=sub.add_parser('report');rep.add_argument('--inputs',type=Path,required=True);rep.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args) if args.command=='prepare' else report(args)


if __name__=='__main__':main()
