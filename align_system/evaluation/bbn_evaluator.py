import pandas as pd

from align_system.evaluation.itm_dataset import ITMDataset


def load_samples(bbn_csv_file):
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
                        kdma_name: kdma_value
                    }
                }
            ]
        }
    ]
    '''
    df = pd.read_csv(bbn_csv_file)
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
    

def load_dataset(bbn_csv_file):
    inputs = []
    labels = []
    for sample in load_samples(bbn_csv_file):
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
        scenario,
        state,
        probe,
        choices: [
            choice_text,
            ... num_choices
        ]
    }
    
    label = [
        {
            kdma_name: kdma_value,
            ... num_kdmas
        },
        ... num_choices
    ]
    '''
    
    return ITMDataset(inputs, labels)