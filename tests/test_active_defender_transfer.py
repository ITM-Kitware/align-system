import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_classifier_replay import request,REPLAY
from test_blue_prompt_screen import TABLE

ROOT=Path(__file__).resolve().parents[1]


class ActiveTransferTest(unittest.TestCase):
    def test_cli_preserves_prompts_excludes_examples_and_reports_paired_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['monitor','active','screen-inputs','screen-output']:(root/name).mkdir()
            seeds=[100,101,*range(110,120),102,*range(120,127)]
            partitions={'examples':[100,115,117,118],
                'development':[121,120,111,122,102,112,124,110],
                'confirmation':[125,101,116,114,119,126,123,113]}
            (root/'screen-inputs/protocol.json').write_text(json.dumps({'shared_seed_partitions':partitions}))
            (root/'screen-output/selection.json').write_text(json.dumps({'selected_variants':['P4','P6']}))
            template=request('template');template['variant']='P4';template['messages']=[
                {'role':'system','content':'FROZEN_BLUE_EXAMPLES'},
                {'role':'user','content':'# OBSERVATIONS\nold\n\n# AVAILABLE LABELS\nfixed P4 labels'}]
            (root/'screen-inputs/development-requests.jsonl').write_text(json.dumps(template)+'\n')
            episodes={'monitor':[],'active':[]};requests=[]
            for label in REPLAY.LABELS:
                directory=label.lower().replace('b-line','bline').replace('-','_')
                for index,seed in enumerate(seeds):
                    name=f'{directory}-{seed}.json';trajectory=f'SECRET_IDENTITY/{name}'
                    common={'seed':seed,'scenario_sha256':'scenario','cage_python_source_sha256':'source','cage_revision':'revision'}
                    monitor={**common,'steps':[{'step':i,'blue_action':{'Action':'Monitor','Status':'TRUE'},
                        'defender_observation':TABLE,'red_action':'SECRET_RED_ACTION','reward':'SECRET_REWARD','metric_scores':'SECRET_METRIC'} for i in range(100)]}
                    path=root/'monitor'/name;path.write_text(json.dumps(monitor));monitor_sha=hashlib.sha256(path.read_bytes()).hexdigest()
                    meta={'class_directory':directory,'trajectory_id':trajectory,'seed':seed,'capture_path':name,
                          'capture_sha256':monitor_sha,'fresh_process_repeat_identical':True,'historical_initial_network_matches':True,
                          'historical_metric_scores_match':True,'historical_action_trace_matches':True}
                    episodes['monitor'].append(meta)
                    active=copy.deepcopy(monitor);active.update(training_seed=index%5,checkpoint_sha256='SECRET_CHECKPOINT',
                        monitor_capture_sha256=monitor_sha,repeat_capture_identical=True,monitor_control_reproduced=True,initial_network_matches_monitor=True)
                    for row in active['steps']:row['blue_action']={'Action':'Restore','Host':'Database0','Status':'TRUE'}
                    path=root/'active'/name;path.write_text(json.dumps(active))
                    episodes['active'].append({**meta,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_seed':index%5})
                    for end in [17,37,58,79,100]:
                        base=request(f'{directory}-{seed}-{end}');base.update(true_label=label,trajectory_id=trajectory,
                            provenance={'class_directory':directory,'window_start':end-7,'window_end':end})
                        base['messages']=[{'role':'system','content':'FROZEN_LEGACY'},
                            {'role':'user','content':'# OBSERVATIONS\nold\n\n# AVAILABLE LABELS\nfixed legacy labels'}]
                        requests.append(base)
            (root/'monitor/summary.json').write_text(json.dumps({'status':'complete','episodes':episodes['monitor']}))
            monitor_summary_sha=hashlib.sha256((root/'monitor/summary.json').read_bytes()).hexdigest()
            (root/'active/summary.json').write_text(json.dumps({'status':'complete','condition':'active_hybrid_ppo','episodes':episodes['active'],
                'monitor_audit_sha256':monitor_summary_sha}))
            (root/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            command=[sys.executable,str(ROOT/'scripts/active_defender_transfer.py'),'prepare','--requests',str(root/'requests.jsonl'),
                '--screen-inputs',str(root/'screen-inputs'),'--screen-output',str(root/'screen-output'),
                '--monitor-captures',str(root/'monitor'),'--active-captures',str(root/'active'),'--output',str(root/'inputs')]
            result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(l) for l in (root/'inputs/requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1440)
            self.assertEqual(len([r for r in rows if r['cohort']=='reserved']),720)
            for row in rows:
                self.assertNotIn(row['seed'],partitions['examples'])
                text=json.dumps(row['messages'])
                for hidden in ['SECRET_RED_ACTION','SECRET_REWARD','SECRET_METRIC','SECRET_CHECKPOINT','SECRET_IDENTITY']:self.assertNotIn(hidden,text)
                self.assertEqual(row['messages'][0]['content'],'FROZEN_LEGACY' if row['variant']=='active_P0' else 'FROZEN_BLUE_EXAMPLES')
                self.assertTrue(row['messages'][-1]['content'].endswith('fixed legacy labels' if row['variant']=='active_P0' else 'fixed P4 labels'))
                self.assertEqual(row['generation']['max_tokens'],256)
            protocol=json.loads((root/'inputs/protocol.json').read_text())
            self.assertEqual(set(protocol['checkpoint_episode_coverage']['all_non_example']),set('01234'))
            self.assertNotIn('2',protocol['checkpoint_episode_coverage']['reserved'])
            output=root/'output';(output/'inference').mkdir(parents=True)
            predicted=[{'request_id':r['request_id'],'true_label':r['true_label'],'predicted_label':r['true_label'],
                        'inference_seconds':1,'input_tokens':100} for r in rows]
            (output/'inference/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predicted))
            reported=subprocess.run([sys.executable,str(ROOT/'scripts/active_defender_transfer.py'),'report',
                '--inputs',str(root/'inputs'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(reported.returncode,0,reported.stderr)
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(report['cohorts']['reserved']['metrics']['active_P4']['n'],240)
            self.assertEqual(report['cohorts']['all_non_example']['metrics']['active_P4']['n'],480)
            self.assertEqual(report['cohorts']['reserved']['joint_accuracy_contrasts']['active_P4 minus monitor_P4']['joint_accuracy_difference'],0)
            reserved=[r for r in rows if r['cohort']=='reserved']
            (output/'selected-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reserved))
            ids={r['request_id'] for r in reserved}
            (output/'inference/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predicted if r['request_id'] in ids))
            smaller=subprocess.run([sys.executable,str(ROOT/'scripts/active_defender_transfer.py'),'report',
                '--inputs',str(root/'inputs'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(smaller.returncode,0,smaller.stderr)
            small_report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(small_report['cohorts']['reserved']['metrics']['active_P4']['status'],'complete')
            self.assertEqual(small_report['cohorts']['all_non_example']['metrics']['active_P4']['status'],'partial')
            self.assertEqual(small_report['cohorts']['all_non_example']['metrics']['active_P4']['expected'],480)
            path=root/'active'/episodes['active'][0]['capture_path'];path.write_text(path.read_text()+' ')
            command[-1]=str(root/'bad')
            bad=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
