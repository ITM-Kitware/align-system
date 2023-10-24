import pandas as pd

from align_system.evaluation.adm_evaluator import ADMEvaluator
from align_system.evaluation.itm_dataset import ITMDataset


# Evaluator sub-classes implement all the BBN-dataset-specific logic to get the data into the right format
class BBNEvaluator(ADMEvaluator):
    
    def __init__(self, bbn_csv_file):
        self.bbn_csv_file = bbn_csv_file
        inputs = []
        labels = []
        for sample in self.load_samples():
            inputs.append({
                key: value
                for key, value in sample.items() if key != 'choices'
            })
            inputs[-1]['choices'] = [
                choice['text']
                for choice in sample['choices']
            ]
            labels.append([
                choice['kdmas']
                for choice in sample['choices']
            ])
            # delete kdmas from choices
            for choice in sample['choices']:
                del choice['kdmas']
                
        '''
        input = {
            scenario_id,
            probe_id,
            scenario,
            state, 
            probe,
            choices: [
                choice_text,
                ...
            ]
        }
        
        label = [
            {
                kdma_name: kdma_value,
                ...
            },
            ... num_choices
        ]
        '''
        
        self._dataset = ITMDataset(inputs, labels)
    
    
    @property
    def dataset(self):
        return self._dataset
    
                
    def load_samples(self):
        '''
        samples = [
            {
                scenario_id,
                probe_id,
                scenario,
                state, 
                probe,
                choices: [
                    {
                        text,
                        kdmas: {
                            basic_knowledge,
                            time_pressure,
                            risk_aversion,
                            fairness,
                            protocol_focus,
                            utilitarianism
                        }
                    }
                ]
            }
        ]
        '''
        df = pd.read_csv(self.bbn_csv_file)
        samples = []
        for probe_id, probe_df in df.groupby('probe_id'):
            sample = None
            for i, row in probe_df.iterrows():
                if sample is None:
                    sample = {
                        'scenario_id': row['scenario_id'],
                        'probe_id': row['probe_id'],
                        'scenario': row['scenario'],
                        'state': row['state'] if not pd.isna(row['state']) else None,
                        'probe': row['probe']
                    }
                    sample['choices'] = []
                    samples.append(sample)
                
                sample['choices'].append({
                    'text': row['answer'],  
                    'kdmas': {
                        kdma: row[kdma]
                        for kdma in ['basic_knowledge', 'time_pressure', 'risk_aversion', 'fairness', 'protocol_focus', 'utilitarianism']
                        if not pd.isna(row[kdma])
                    }
                })

        return samples