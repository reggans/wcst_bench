import numpy as np
import json

def score(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    scores = []
    for trial in data:
        current_rule = ""
        completion_lengths = []

        for query in trial:
            if query['rule'] != current_rule:
                current_rule = query['rule']
                completion_lengths.append(0)
            else:
                completion_lengths[-1] += 1
        
        scores.append(np.mean(completion_lengths))
    return np.mean(scores)