import json, os
from random import random


def load_json(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def compute_similarity(record, memory_list, kcg, know_course_list):

    sim = []
    rec = record[1].lower().strip().replace('\"','')
    rec_id = know_course_list.get(rec)
    for m in memory_list:
        mem = m[1].lower().strip().replace('\"','')
        mem_id = know_course_list.get(mem)
        if (rec_id, mem_id) in kcg or (mem_id, rec_id) in kcg:
            sim.append(1)
        else:
            if rec_id == mem_id and random.random() > 0.8:
                sim.append(1)
            else:
                sim.append(0)
    return sim
