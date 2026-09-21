import json,collections
c=collections.Counter(); s=[]
for l in open('labels_vacation_hybrid_targeted_test_noleak.jsonl'):
    r=json.loads(l); c[r['label_source']]+=1
    if r['label_source']=='teacher_hybrid':
        s.append(json.loads(r['raw_response'])['gdino_score'])
print(c); print('gdino scores:', min(s), max(s), len(s))