#!/usr/bin/env python3
"""Report paired follow-up comparisons without inference dependencies."""

import argparse
import json
from pathlib import Path

from blue_prompt_screen_report import score, predictions, grouped_difference
from run_classifier_replay import read_requests, write_json


def build(inputs,output):
    result={'protocol':json.loads((inputs/'protocol.json').read_text()),'stages':{}}
    for stage in ['development','confirmation','fresh']:
        request_path=output/f'{stage}-requests.jsonl'
        if not request_path.exists():continue
        requests=read_requests(request_path)
        completed=predictions(output/stage/'predictions.jsonl')
        if set(completed)-{r['request_id'] for r in requests}:raise ValueError('Foreign saved predictions')
        variants=sorted({r['variant'] for r in requests})
        metrics={v:score([r for r in requests if r['variant']==v],completed) for v in variants}
        pairs=[]
        for candidate,reference in [('R-short','T-short'),('R-prefix','T-prefix'),
                                    ('T-prefix','T-short'),('R-prefix','R-short')]:
            if candidate in variants and reference in variants:pairs.append((candidate,reference))
        if 'P0' in variants:pairs.extend((v,'P0') for v in variants if v!='P0')
        contrasts={}
        for candidate,reference in pairs:
            comparator={r['pair_id']:completed[r['request_id']] for r in requests
                        if r['variant']==reference and r['request_id'] in completed}
            contrasts[candidate+' minus '+reference]=grouped_difference(
                [r for r in requests if r['variant']==candidate],completed,comparator)
            if stage=='fresh' and contrasts[candidate+' minus '+reference] is not None:
                contrasts[candidate+' minus '+reference]['interpretation']='Frozen comparison on newly seeded episodes; within-protocol generalization only.'
        result['stages'][stage]={'variants':metrics,'paired_joint_accuracy_differences':contrasts,
            'shared_seed_groups':len({r['seed'] for r in requests}),
            'by_prediction_time':{str(end):{v:score([r for r in requests if r['variant']==v and r['provenance']['window_end']==end],completed)
                for v in variants} for end in sorted({r['provenance']['window_end'] for r in requests})}}
    for name in ['budget-decision.json','selection.json','fresh-budget.json','context-audit.json']:
        if (output/name).exists():
            data=json.loads((output/name).read_text());data.pop('input_tokens_by_request',None)
            result[name.removesuffix('.json')]=data
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'comparison.json',result)
    lines=['# Overnight classifier follow-up results','','Partial conditions remain partial. Shared rollout seeds are the replication unit.','',
           '| Stage | Variant | Status | Requests | Joint | Family | Objective |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for stage,data in result['stages'].items():
        for variant,m in data['variants'].items():
            values=[f"{100*m[k]:.2f}%" if m[k] is not None else '' for k in ['joint_accuracy','family_accuracy','objective_accuracy']]
            lines.append(f"| {stage} | {variant} | {m['status']} | {m['n']}/{m['expected']} | "+' | '.join(values)+' |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args.inputs,args.output)
    print(json.dumps({'report':str(args.output/'REPORT.md')}))


if __name__=='__main__':main()
