from abc import abstractmethod

# Implements the metric logic
class ADMEvaluator:
    
    @property
    @abstractmethod
    def dataset(self):
        pass
    
    
    def generate_outputs(self, adm, target_kdmas, **kwargs):
        outputs = []
        for input_, label in self.dataset:
            # add target kdmas to input without changing the dataset
            input_ = input_.copy()
            input_['target_kdmas'] = target_kdmas
            outputs.append(adm(input_, **kwargs))
            # use dummy data
            # outputs.append({
            #     'choice': 0,
            #     'predicted_kdmas': [
            #         {kdma_name: 0 for kdma_name in target_kdmas}
            #         for _ in range(len(input_['choices']))
            #     ]
            # })
        
        return outputs
    
    
    def evaluate(self, generated_outputs):
        
        system_kdmas = self.get_avg_system_kdmas(generated_outputs)
        
        metrics = [
            ADMEvaluator.mean_absolute_error,
            ADMEvaluator.mean_squared_error,
            ADMEvaluator.soartech_similarity_score,
            ADMEvaluator.soartech_similarity_score_by_kdma,
            ADMEvaluator.adept_similarity_score,
            ADMEvaluator.adept_similarity_score_by_kdma,
        ]
        
        results = {
            'choice_metrics': {
                'system_kdmas': system_kdmas,
            },
        }
        
        for metric in metrics:
            metric_name = metric.__name__
            results['choice_metrics'][metric_name] = metric(target_kdmas, system_kdmas)
            
        metrics = [
            ADMEvaluator.mean_absolute_error,
            ADMEvaluator.mean_squared_error,
            ADMEvaluator.soartech_similarity_score,
            ADMEvaluator.adept_similarity_score,
        ]
        
        per_choice_metrics = []
            
        for output, (input_, label) in zip(generated_outputs, self.dataset):
            if not 'predicted_kdmas' in output:
                continue
            
            
            for label_kdmas, predicted_kdmas in zip(label, output['predicted_kdmas']):
                choice_metrics = {}
                for metric in metrics:
                    metric_name = metric.__name__
                    choice_metrics[metric_name] = metric(label_kdmas, predicted_kdmas)
            
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
    
    
    def get_avg_system_kdmas(self, outputs):
        chosen_kdmas = {}
        for output, (input_, label) in zip(outputs, self.dataset):
            choice_idx = output['choice']
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

    @staticmethod
    def adept_similarity_score(target_kdmas, system_kdmas):
        if len(target_kdmas) == 0:
            return 0
        
        distance = 0
        for kdma, target_value in target_kdmas.items():
            system_value = system_kdmas[kdma] if kdma in system_kdmas else 5
            distance += (target_value - system_value) ** 2
        
        return 1 / (distance + 1)
    
    
    @staticmethod
    def adept_similarity_score_by_kdma(target_kdmas, system_kdmas):
        if len(target_kdmas) == 0:
            return {}
        
        scores = {}
        for kdma, target_value in target_kdmas.items():
            system_value = system_kdmas[kdma] if kdma in system_kdmas else 5
            distance = (target_value - system_value) ** 2
            scores[kdma] = 1 / (distance + 1)
        
        return scores
    
    
    @staticmethod
    def soartech_similarity_score(target_kdmas, system_kdmas, p=2):
        kdmas = set(target_kdmas.keys()) & set(system_kdmas.keys())
        
        if len(kdmas) == 0:
            return 0
        
        a = [target_kdmas[kdma]/10 for kdma in kdmas]
        b = [system_kdmas[kdma]/10 for kdma in kdmas]
        
        for vec in (a,b):
            assert min(vec) >= 0
            assert max(vec) <= 1
        
        return 1 - sum([(abs(ai - bi)**p)   for ai, bi in zip(a,b)])/len(kdmas)
    
    
    @staticmethod
    def soartech_similarity_score_by_kdma(target_kdmas, system_kdmas, p=2):
        kdmas = set(target_kdmas.keys()) & set(system_kdmas.keys())
        
        if len(kdmas) == 0:
            return {}
        
        a = [target_kdmas[kdma]/10 for kdma in kdmas]
        b = [system_kdmas[kdma]/10 for kdma in kdmas]
        
        for vec in (a,b):
            assert min(vec) >= 0
            assert max(vec) <= 1
        
        return {kdma: 1 - (abs(ai - bi)**p) for kdma, ai, bi in zip(kdmas, a, b)}

    
    @staticmethod
    def mean_absolute_error(target_kdmas, system_kdmas):
        kdmas = set(target_kdmas.keys()) & set(system_kdmas.keys())
        
        if len(kdmas) == 0:
            return 0
        
        a = [target_kdmas[kdma] for kdma in kdmas]
        b = [system_kdmas[kdma] for kdma in kdmas]
        
        return sum([abs(ai - bi)   for ai, bi in zip(a,b)])/len(kdmas)
    
    
    @staticmethod
    def mean_squared_error(target_kdmas, system_kdmas):
        kdmas = set(target_kdmas.keys()) & set(system_kdmas.keys())
        
        if len(kdmas) == 0:
            return 0
        
        a = [target_kdmas[kdma] for kdma in kdmas]
        b = [system_kdmas[kdma] for kdma in kdmas]
        
        return sum([(ai - bi)**2   for ai, bi in zip(a,b)])/len(kdmas)