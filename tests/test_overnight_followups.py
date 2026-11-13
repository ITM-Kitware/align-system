import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_classifier_replay import request
from test_blue_prompt_screen import TABLE

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from legal_event_history import raw_history
from run_overnight_followups import choose_fresh


class FollowupTest(unittest.TestCase):
    def test_raw_host_evidence_round_trips_with_order_repetitions_and_missing_keys(self):
        connection={'local_address':'10.0.0.1','local_port':22,'remote_address':'10.0.0.2','remote_port':4444,'transport':'tcp'}
        raw={'Database0':{'System info':{'Hostname':'Database0'},'Interface':[],
             'Processes':[{'PID':9,'Connections':[connection,{'local_port':80},connection]}],'Files':[]},'success':'TRUE'}
        capture={'steps':[{'defender_observation':TABLE,'defender_raw_observation':raw},
                          {'defender_observation':TABLE,'defender_raw_observation':raw}]}
        encoded=json.loads(raw_history(capture,0,2));previous={};restored=[]
        for _,hosts in encoded:
            decoded={}
            for row in hosts:
                attrs=previous.setdefault(row['host'],{});attrs.update(row['attribute_updates'])
                host={k:attrs[k] for k in row['attributes_present']}
                host['Files']=row['files'];host['Processes']=[]
                for process in row['processes']:
                    original=copy.deepcopy(process['fields']);original['Connections']=[]
                    for fields in process['connections']:
                        conn=copy.deepcopy(fields[4])
                        for i,key in enumerate(['local_address','local_port','remote_address','remote_port']):
                            if fields[5]&(1<<i):conn[key]='10.0.0.1' if fields[i]=='@Database0' else fields[i]
                        original['Connections'].append(conn)
                    host['Processes'].append(original)
                decoded[row['host']]=host
            restored.append(decoded)
        self.assertEqual(restored,[{'Database0':raw['Database0']}]*2)
        contaminated=copy.deepcopy(capture);contaminated['steps'][0]['defender_raw_observation']['Database0']['red_action']='hidden'
        with self.assertRaisesRegex(ValueError,'unreviewed'):raw_history(contaminated,0,2)

    def test_fresh_preparation_cli_preserves_frozen_systems_and_rejects_changed_capture(self):
        labels=['B-line-C','B-line-I','B-line-A','Meander-C','Meander-I','Meander-A']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);captures=root/'captures';captures.mkdir();screen=root/'screen';screen.mkdir();primary=root/'primary';primary.mkdir()
            base=request('old');base['messages']=[{'role':'system','content':'FROZEN_PRIMARY'},{'role':'user','content':'# OBSERVATIONS\nold\n\n# AVAILABLE LABELS\n'+'\n'.join(labels)}]
            base.update(variant='P3',provenance={'window_start':30,'window_end':37},input_view='defender_table_compact_history')
            (screen/'development-requests.jsonl').write_text(json.dumps(base)+'\n')
            legacy=copy.deepcopy(base);legacy.pop('variant');legacy['messages'][0]['content']='FROZEN_LEGACY'
            legacy['generation']['max_tokens']=32
            (root/'legacy.jsonl').write_text(json.dumps(legacy)+'\n');(primary/'selection.json').write_text(json.dumps({'selected_variants':['P3']}))
            episodes=[]
            for seed in [20000,20001]:
                for label in labels:
                    directory=label.lower().replace('b-line','bline').replace('-','_');name=f'{directory}-{seed}.json'
                    capture={'seed':seed,'repeat_capture_identical':True,'cage_python_source_sha256':'fixed',
                        'steps':[{'step':i,'blue_action':{'Action':'Monitor','Status':'TRUE'},'defender_observation':TABLE,
                                  'defender_raw_observation':{'success':'TRUE'},'red_action':'SECRET_RED_ACTION','metric_scores':'SECRET_METRICS'} for i in range(100)]}
                    path=captures/name;path.write_text(json.dumps(capture))
                    episodes.append({'seed':seed,'true_label':label,'class_directory':directory,'trajectory_id':'SECRET_IDENTITY/'+name,
                                     'capture_path':name,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            (captures/'summary.json').write_text(json.dumps({'status':'complete','condition':'fresh_monitor','expected_episodes':12,
                'seed_groups':[20000,20001],'cage_python_source_sha256':'fixed','episodes':episodes}))
            cmd=[sys.executable,str(ROOT/'scripts/prepare_overnight_followups.py'),'fresh','--screen-inputs',str(screen),
                 '--capture-dir',str(captures),'--output',str(root/'inputs'),'--primary-output',str(primary),'--legacy-requests',str(root/'legacy.jsonl')]
            result=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(l) for l in (root/'inputs/fresh-requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),120)
            for row in rows:
                self.assertEqual(row['messages'][0]['content'],'FROZEN_LEGACY' if row['variant']=='P0' else 'FROZEN_PRIMARY')
                self.assertEqual(row['actual_history_length'],row['provenance']['window_end']-row['provenance']['window_start'])
                self.assertEqual(row['generation']['max_tokens'],256)
                for secret in ['SECRET_RED_ACTION','SECRET_METRICS','SECRET_IDENTITY']:self.assertNotIn(secret,json.dumps(row['messages']))
            path=captures/episodes[0]['capture_path'];path.write_text(path.read_text()+' ')
            cmd[cmd.index('--output')+1]=str(root/'bad')
            bad=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0)
            self.assertFalse((root/'bad/fresh-requests.jsonl').exists())

    def test_budget_keeps_full_balanced_seed_groups_without_consulting_scores(self):
        rows=[{'seed':seed,'variant':variant,'provenance':{'window_end':end}} for seed in range(100)
              for variant in ['P0','P3'] for end in [17,37,58,79,100] for _ in range(6)]
        selected,budget=choose_fresh(rows,3,14000)
        self.assertGreaterEqual(budget['shared_seed_groups'],50)
        self.assertFalse(budget['fresh_accuracy_used_for_budget'])
        self.assertEqual(len(selected),budget['shared_seed_groups']*len(budget['prediction_times'])*12)
        with self.assertRaisesRegex(ValueError,'Insufficient time'):choose_fresh(rows,30,1000)


if __name__=='__main__':unittest.main()
