import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_classifier_replay import request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_blue_prompt_screen import compact_history, LABELS
from align_system.interfaces.cage_observations import format_window


TABLE='''+---+
| Subnet | IP Address | Hostname | Activity | Compromised |
| 10.0.0.0/24 | 10.0.0.1 | Database0 | Scan | No |
+---+'''


class BluePromptScreenTest(unittest.TestCase):
    def test_compact_history_retains_repeated_activity_and_action_results(self):
        capture={'steps':[{'step':i,'defender_observation':TABLE,
                           'blue_action':{'Action':'Monitor','Status':'TRUE'}} for i in range(3)]}
        text=compact_history(capture,0,3)
        self.assertEqual(text.count('"Scan"'),3)
        self.assertEqual(text.count('Blue Monitor (success: TRUE)'),3)
        capture['steps'][2]['defender_observation']=TABLE.replace('Scan','None')
        text=compact_history(capture,0,3)
        self.assertIn('"Database0","None","No"',text)

    def test_cli_separates_shared_seeds_and_keeps_hidden_fields_out_of_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);captures=root/'captures';captures.mkdir()
            episodes=[];requests=[];predictions=[]
            for label in LABELS:
                directory=label.lower().replace('b-line','bline').replace('-','_')
                for seed in range(100,120):
                    trajectory=f'SECRET_TRAJECTORY_{label}/E{seed}.json'
                    capture={'seed':seed,'steps':[{
                        'step':i,'defender_observation':TABLE,'blue_action':{'Action':'Monitor','Status':'TRUE'},
                        'red_action':{'Action':'SECRET_RED_ACTION','Status':'TRUE'},
                        'defender_raw_observation':{'success':True,'Database0':{'Processes':[]}},
                        'metric_scores':{'SECRET_METRIC':42},
                    } for i in range(100)]}
                    path=captures/f'{directory}-{seed}.json'
                    path.write_text(json.dumps(capture))
                    episodes.append({'class_directory':directory,'trajectory_id':trajectory,'seed':seed,
                        'capture_path':path.name,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'fresh_process_repeat_identical':True,'historical_initial_network_matches':True,
                        'historical_metric_scores_match':True,'historical_action_trace_matches':True})
                    for start,end in [(10,17),(30,37),(51,58),(72,79),(93,100)]:
                        row=request(f'{directory}-{seed}-{end}')
                        row.update(true_label=label,trajectory_id=trajectory,
                            provenance={'class_directory':directory,'window_start':start,'window_end':end,'window_size':7},
                            messages=[{'role':'system','content':'SECRET_OLD_RED_EXAMPLES'},
                                      {'role':'user','content':'# OBSERVATIONS\n'+format_window(capture,start,end)+'\n\n# AVAILABLE LABELS\n'+'\n'.join(LABELS)}])
                        requests.append(row)
                        predictions.append({'request_id':row['request_id'],'true_label':label,
                                            'predicted_label':label,'inference_seconds':1})
            (captures/'summary.json').write_text(json.dumps({'status':'complete','episodes':episodes}))
            for name,rows in [('requests',requests),('predictions',predictions)]:
                (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            command=[sys.executable,str(ROOT/'scripts/prepare_blue_prompt_screen.py'),
                '--requests',str(root/'requests.jsonl'),'--baseline-predictions',str(root/'predictions.jsonl'),
                '--capture-dir',str(captures),'--output-dir',str(root/'inputs')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            split=json.loads((root/'inputs/protocol.json').read_text())['shared_seed_partitions']
            self.assertEqual({k:len(v) for k,v in split.items()}, {'examples':4,'development':8,'confirmation':8})
            for partition,expected in [('development',480),('confirmation',1200)]:
                rows=[json.loads(line) for line in (root/f'inputs/{partition}-requests.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),expected)
                for row in rows:
                    text=json.dumps(row['messages'])
                    for hidden in ['SECRET_RED_ACTION','SECRET_METRIC','SECRET_OLD_RED_EXAMPLES','SECRET_TRAJECTORY']:
                        self.assertNotIn(hidden,text)
                    self.assertIn(row['seed'],split[partition])
                    self.assertTrue(set(row['provenance']['example_episode_seeds']) <= set(split['examples']))
                    self.assertNotIn(row['seed'],row['provenance']['example_episode_seeds'])
                    if row['variant']=='P4':
                        self.assertEqual(row['messages'][0]['content'].count('Training example '),12)
            repeated=root/'repeat'
            command[-1]=str(repeated)
            again=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(again.returncode,0,again.stderr)
            self.assertEqual((repeated/'development-requests.jsonl').read_bytes(),(root/'inputs/development-requests.jsonl').read_bytes())
            capture_path=captures/episodes[0]['capture_path']
            capture_path.write_text(capture_path.read_text()+' ')
            command[-1]=str(root/'bad')
            changed=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':
    unittest.main()
