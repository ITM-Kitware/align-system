#!/usr/bin/env python3
"""Collect independently seeded, repeated Monitor episodes for frozen evaluation."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys

from capture_defender_observations import json_safe, simulator_source_digest
from run_classifier_replay import LABELS, digest, write_json


def collect(seed, label, scenario, steps):
    import numpy as np
    from CybORG import CybORG
    from CybORG.Agents import GreenAgent
    from CybORG.Agents.SimpleAgents.B_line_resilience import BlineJumping_C, BlineJumping_I, BlineJumping_A
    from CybORG.Agents.SimpleAgents.Meander_Resilience import MeanderJumping_C, MeanderJumping_I, MeanderJumping_A
    from CybORG.Agents.Wrappers import BlueTableWrapper
    from CybORG.Shared.Actions import Monitor

    attackers=dict(zip(LABELS,(BlineJumping_C,BlineJumping_I,BlineJumping_A,
                              MeanderJumping_C,MeanderJumping_I,MeanderJumping_A)))
    random.seed(seed);np.random.seed(seed)
    env=CybORG(str(scenario),'sim',agents={'Red':attackers[label],'Green':GreenAgent})
    env.set_seed(seed)
    random.seed(seed);np.random.seed(seed)
    wrapper=BlueTableWrapper(env,output_mode='table')
    initial=str(wrapper.reset(agent='Blue').observation)
    rows=[]
    for index in range(steps):
        result=wrapper.step(agent='Blue',action=Monitor(agent='Blue',session=0))
        observation=env.get_observation('Blue')
        rows.append({'step':index,'defender_observation':str(result.observation),
                     'defender_raw_observation':json_safe(observation),
                     'blue_action':{'Action':'Monitor','Host':'Monitor','Status':str(observation['success'])}})
        if result.done and index+1 != steps:
            raise ValueError('Fresh episode ended before the recorded horizon')
    return {'seed':seed,'initial_defender_observation':initial,'steps':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cage-root',required=True,type=Path)
    p.add_argument('--cage-revision',required=True)
    p.add_argument('--scenario',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--start-seed',type=int,default=20000)
    p.add_argument('--seed-count',type=int,default=50)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--excluded-protocol',required=True,type=Path)
    args=p.parse_args()
    protocol=json.loads(args.excluded_protocol.read_text())
    excluded={seed for values in protocol['shared_seed_partitions'].values() for seed in values}
    seeds=list(range(args.start_seed,args.start_seed+args.seed_count))
    if excluded.intersection(seeds) or args.seed_count<1 or args.steps<100:
        raise ValueError('Fresh seed groups must be disjoint and cover the declared horizon')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.cage_root.resolve()))
    fingerprint=simulator_source_digest(args.cage_root)
    episodes=[]
    write_json(args.output/'summary.json',{'status':'partial','condition':'fresh_monitor',
               'expected_episodes':len(seeds)*6,'episodes':[],'seed_groups':seeds})
    for seed in seeds:
        for label in LABELS:
            first=collect(seed,label,args.scenario,args.steps)
            second=collect(seed,label,args.scenario,args.steps)
            if first != second:
                write_json(args.output/f'failed-repeat-{seed}-{label}.json',{'first':first,'second':second})
                raise ValueError(f'Repeated fresh capture changed for seed {seed}, class {label}')
            directory=label.lower().replace('b-line','bline').replace('-','_')
            path=args.output/f'{directory}-{seed}.json'
            capture={**first,'condition':'fresh_monitor','repeat_capture_identical':True,
                     'cage_revision':args.cage_revision,'cage_python_source_sha256':fingerprint,
                     'scenario_sha256':digest(args.scenario),'collector_sha256':digest(Path(__file__)),
                     'observation_timing':'post-action; step 0 follows first Monitor action',
                     'observation_source':'Blue observations only; no historical-reproduction claim'}
            write_json(path,capture)
            episodes.append({'class_directory':directory,'true_label':label,'seed':seed,
                             'capture_path':path.name,'capture_sha256':digest(path),
                             'trajectory_id':f'fresh/{directory}/E{seed}.json'})
            write_json(args.output/'summary.json',{'status':'partial','condition':'fresh_monitor',
                       'expected_episodes':len(seeds)*6,'episodes':episodes})
            print(json.dumps({'completed_episodes':len(episodes),'expected':len(seeds)*6}),flush=True)
    if simulator_source_digest(args.cage_root) != fingerprint:
        raise ValueError('Simulator changed during fresh capture')
    write_json(args.output/'summary.json',{'status':'complete','condition':'fresh_monitor',
               'expected_episodes':len(seeds)*6,'episodes':episodes,'seed_groups':seeds,
               'cage_revision':args.cage_revision,'cage_python_source_sha256':fingerprint,
               'scenario_sha256':digest(args.scenario),'collector_sha256':digest(Path(__file__)),
               'excluded_protocol_sha256':digest(args.excluded_protocol),
               'python_version':sys.version,'software':{name:importlib.metadata.version(name) for name in
                   ['numpy','gymnasium','prettytable','PyYAML','paramiko']},
               'interpretation':'New seeded episodes, each repeated exactly; not historical trajectory reproductions.'})


if __name__=='__main__':
    main()
