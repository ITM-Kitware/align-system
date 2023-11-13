

def generate_outputs(dataset, adm, target_kdma_values, **kwargs):
    outputs = []
    for input_, label in dataset:
        # add target kdmas to input without changing the dataset
        outputs.append(adm(input_, target_kdma_values, labels=label, **kwargs))
    
    return outputs


def get_avg_system_kdmas(dataset, outputs):
    chosen_kdmas = {}
    for output, (input_, label) in zip(outputs, dataset):
        choice_idx = output['choice']
        if choice_idx is None:
            continue
        
        label_kdmas = label[choice_idx]
        for kdma_name, kdma_value in label_kdmas.items():
            if kdma_name not in chosen_kdmas:
                chosen_kdmas[kdma_name] = []
            chosen_kdmas[kdma_name].append(kdma_value)
        
    avg_kdmas = {
        kdma_name: sum(kdma_values) / (len(kdma_values) + 1e-9)
        for kdma_name, kdma_values in chosen_kdmas.items()
    }
    
    return avg_kdmas


def adept_similarity_score(target_kdma_values, system_kdmas):
    if len(target_kdma_values) == 0:
        return 0
    
    distance = 0
    for kdma, target_value in target_kdma_values.items():
        system_value = system_kdmas[kdma] if kdma in system_kdmas else 5
        distance += (target_value - system_value) ** 2
    
    return 1 / (distance + 1)


def adept_similarity_score_by_kdma(target_kdma_values, system_kdmas):
    if len(target_kdma_values) == 0:
        return {}
    
    scores = {}
    for kdma, target_value in target_kdma_values.items():
        system_value = system_kdmas[kdma] if kdma in system_kdmas else 5
        distance = (target_value - system_value) ** 2
        scores[kdma] = 1 / (distance + 1)
    
    return scores


def soartech_similarity_score(target_kdma_values, system_kdmas, p=0.75):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return 0
    
    a = [target_kdma_values[kdma]/10 for kdma in kdmas]
    b = [system_kdmas[kdma]/10 for kdma in kdmas]
    
    for vec in (a,b):
        assert min(vec) >= 0
        assert max(vec) <= 1
    
    return 1 - sum([(abs(ai - bi)**p)   for ai, bi in zip(a,b)])/len(kdmas)


def kitware_similarity_score(target_kdma_values, system_kdmas):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return 0
    
    a = [target_kdma_values[kdma] for kdma in kdmas]
    b = [system_kdmas[kdma] for kdma in kdmas]
    
    for vec in (a,b):
        assert min(vec) >= 0
        assert max(vec) <= 10
    
    return sum([
        10**(1 - (ai - bi)**2/25)/10
        for ai, bi in zip(a,b)
    ])/len(kdmas)


def kitware_similarity_score_by_kdma(target_kdma_values, system_kdmas):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return 0
    
    a = [target_kdma_values[kdma] for kdma in kdmas]
    b = [system_kdmas[kdma] for kdma in kdmas]
    
    for vec in (a,b):
        assert min(vec) >= 0
        assert max(vec) <= 10
    
    return {
        kdma: 10**(1 - (ai - bi)**2/25)/10
        for kdma, ai, bi in zip(kdmas, a, b)
    }
    

    

def soartech_similarity_score_by_kdma(target_kdma_values, system_kdmas, p=0.75):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return {}
    
    a = [target_kdma_values[kdma]/10 for kdma in kdmas]
    b = [system_kdmas[kdma]/10 for kdma in kdmas]
    
    for vec in (a,b):
        assert min(vec) >= 0
        assert max(vec) <= 1
    
    return {kdma: 1 - (abs(ai - bi)**p) for kdma, ai, bi in zip(kdmas, a, b)}


def mean_absolute_error(target_kdma_values, system_kdmas):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return 0
    
    a = [target_kdma_values[kdma] for kdma in kdmas]
    b = [system_kdmas[kdma] for kdma in kdmas]
    
    return sum([abs(ai - bi)   for ai, bi in zip(a,b)])/len(kdmas)


def mean_squared_error(target_kdma_values, system_kdmas):
    kdmas = set(target_kdma_values.keys()) & set(system_kdmas.keys())
    
    if len(kdmas) == 0:
        return 0
    
    a = [target_kdma_values[kdma] for kdma in kdmas]
    b = [system_kdmas[kdma] for kdma in kdmas]
    
    return sum([(ai - bi)**2   for ai, bi in zip(a,b)])/len(kdmas)


def oracle_accuracy(target_kdma_values, label_kdma_values, choice_indecies, alignment_metric=kitware_similarity_score, maximize=True):
    '''
    target_kdma_values: {
        kdma_name: kdma_value,
        ...
    }
    label_kdma_values:[
        [
            {
                kdma_name: kdma_value,
                ...
            },
            ...
        ],
        ...
    ]
    choice_indecies: [
        choice_index,
        ...
    ]
    alignment_metric: function(target_kdma_values, system_kdmas)
    maximize: boolean - whether to maximize or minimize the alignment metric
    '''
    
    oracle_choice_indecies = []
    for label_kdma_value_list in label_kdma_values:
        optimization_fn = max if maximize else min
        oracle_choice_indecies.append(optimization_fn(
            range(len(label_kdma_value_list)),
            key=lambda choice_index: alignment_metric(target_kdma_values, label_kdma_value_list[choice_index])
        ))
    
    correct = 0
    total = 0
    for choice_index, oracle_choice_index in zip(choice_indecies, oracle_choice_indecies):
        total += 1
        if choice_index == oracle_choice_index:
            correct += 1
        
    return correct / total


def evaluate(dataset, generated_outputs, target_kdma_values):
        
    system_kdmas = get_avg_system_kdmas(dataset, generated_outputs)
    
    metrics = [
        mean_absolute_error,
        mean_squared_error,
        soartech_similarity_score,
        soartech_similarity_score_by_kdma,
        adept_similarity_score,
        adept_similarity_score_by_kdma,
        kitware_similarity_score,
        kitware_similarity_score_by_kdma
    ]
    
    results = {
        'choice_metrics': {
            'target_kdmas': target_kdma_values,
            'system_kdmas': system_kdmas,
        },
    }
    
    for metric in metrics:
        metric_name = metric.__name__
        results['choice_metrics'][metric_name] = metric(target_kdma_values, system_kdmas)
    
    oracle_accuracy_value = oracle_accuracy( 
        target_kdma_values,
        [label for _, label in dataset],
        [output['choice'] for output in generated_outputs]
    )
    
    
        
    metrics = [
        mean_absolute_error,
        mean_squared_error,
        soartech_similarity_score,
        adept_similarity_score,
        kitware_similarity_score
    ]
    
    per_choice_metrics = []
        
    for output, (input_, label) in zip(generated_outputs, dataset):
        if not 'predicted_kdma_values' in output:
            continue
        
        for label_kdmas, predicted_kdma_values in zip(label, output['predicted_kdma_values']):
            choice_metrics = {}
            for metric in metrics:
                metric_name = metric.__name__
                choice_metrics[metric_name] = metric(label_kdmas, predicted_kdma_values)
        
            per_choice_metrics.append(choice_metrics)
    
    if len(per_choice_metrics) > 0:
        results['predicted_kdmas_metrics'] = {}
        for metric in metrics:
            metric_name = metric.__name__
            avg_metric_value = sum([
                choice_metrics[metric_name]
                for choice_metrics in per_choice_metrics
            ]) / len(per_choice_metrics)
            results['predicted_kdmas_metrics'][metric_name] = avg_metric_value
    
    return results
    
    
    

    