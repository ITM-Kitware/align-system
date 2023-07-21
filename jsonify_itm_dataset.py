import json
import ast
import pandas as pd
import sys
import os

def convert_csv_to_json(filename):
    # try:
    # Read the CSV file
    df = pd.read_csv(filename)

    # Split the DataFrame by source_dataset
    datasets = df['Source_dataset'].unique()

    output = []

    for dataset in datasets:
        dataset_dict = {'source_dataset': dataset, 'scenarios': []}

        dataset_df = df[df['Source_dataset'] == dataset]
        scenarios = dataset_df['Scenario'].unique()

        print(scenarios)

        for scenario in scenarios:
            scenario_dict = {'scenario': scenario, 'probes': []}

            scenario_df = dataset_df[dataset_df['Scenario'] == scenario]
            probes = scenario_df['Probe'].unique()

            for probe in probes:
                probe_dict = {'probe': probe, 'answers': []}

                probe_df = scenario_df[scenario_df['Probe'] == probe]

                for _, data in probe_df.iterrows():
                    kdma = ast.literal_eval(data['Kdma'])
                    target_kdma = ast.literal_eval(data['Target kdma'])

                    # make keys in kdma snakecase
                    kdma = {k.replace(' ', '_').lower(): v for k, v in kdma.items()}
                    target_kdma = {k.replace(' ', '_').lower(): v for k, v in target_kdma.items()}

                    probe_dict['answers'].append({
                        'answer': data['Answer'],
                        'kdma': kdma,
                        'target_kdma': target_kdma
                    })

                scenario_dict['probes'].append(probe_dict)

            dataset_dict['scenarios'].append(scenario_dict)

        output.append(dataset_dict)
        
    # Define the filename for the resulting JSON
    json_filename = os.path.splitext(filename)[0] + '.json'

    # Write the JSON to the file
    with open(json_filename, 'w') as json_file:
        json.dump(output, json_file, indent=4)

    print(f'Successfully created {json_filename}.')
    # except Exception as e:
    #     print(f'Error: {e}')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python script.py [filename]')
    else:
        convert_csv_to_json(sys.argv[1])