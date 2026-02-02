import json
import os
from collections import defaultdict
from collections import Counter
import numpy
import argparse
from glob import glob

def read_json(path): 
    with open(path) as f:
        data = json.load(f)
    return data

def get_results(data_list):
    tuple_dict = defaultdict(list)
    for data_i in data_list:
        num_q = len(data_i['questions'])
        score_mean1 = sum([1 for q_i in data_i['questions'] if [q_i['answer']+'.'] == q_i['response']]) / num_q
        score_mean2 = sum([1 for q_i in data_i['questions'] if q_i['answer'] == q_i['response']]) / num_q
        score_mean = max(score_mean1, score_mean2)
        tuple_dict[data_i['duration']].append((data_i['url'], score_mean))
    
    return tuple_dict

    
if __name__ == "__main__":
    path_format = "{}/generate_videomme.json"
    target_dirs = glob("results*")
    collect_dict = defaultdict(dict)
    for dir_i in target_dirs:
        target_path = path_format.format(dir_i)
        target_info = get_results(read_json(target_path))
        for k, v in target_info.items():
            for acc_i in v:
                if not acc_i[0] in collect_dict[k]:
                    collect_dict[k][acc_i[0]] = 0
                if float(acc_i[1]) >= 0.9:
                    collect_dict[k][acc_i[0]] += 1
    

    tmp = {k:numpy.mean([int(v_i) for k_i, v_i in v.items()]) for k, v in collect_dict.items()}
    import pdb;pdb.set_trace()
    pass

    score = numpy.mean([numpy.mean([i[1] for i in v]) for k, v in target_info.items()])
    print(target_info_dict)
    print(f"score: {score}")

    import pdb;pdb.set_trace()
    pass

    tuple_dict = defaultdict(list)
    for _type in info_32frames.keys():
        for info32_i in info_32frames[_type]:
            v_id, score_32 = info32_i
            score_64 = dict(info_64frames[_type])[v_id]
            tuple_dict[_type].append(score_64-score_32)
    
    # import pdb;pdb.set_trace()
    # pass
    # {k: numpy.sum(v)for k, v in tuple_dict.items()}
    # {
    #     'short'   : -3.0,                 32 is better
    #     'medium'  : 9.333333333333332,    64 is better
    #     'long'    : 0.33333333333333326   64 is better but similar
    # }

    # ** score_64 - score_32 ** 
    # {k: Counter([round(_, 4) for _ in v])for k, v in tuple_dict.items()}
    # {
    #     'short'   : Counter({0.0: 226, -0.3333: 39, 0.3333: 32, -0.6667: 2, 0.6667: 1}), 
    #     'medium'  : Counter({0.0: 233, -0.3333: 18, 0.3333: 44, -0.6667: 2, 0.6667: 3}), 
    #     'long'    : Counter({0.0: 228, -0.3333: 34, 0.3333: 33, -0.6667: 2, 0.6667: 3})
    # }
