import json
lines = [json.loads(l) for l in open('labels_test.jsonl') if l.strip()]
sample = [r for r in lines if r.get('pred_x') and r.get('gt_x')]
print(f'Records with both pred and GT: {len(sample)}')
if sample:
    print('Sample:', sample[0]['pred_x'], sample[0]['gt_x'])


'''
python -c "
import json
lines = [json.loads(l) for l in open('labels_test.jsonl') if l.strip()]
sample = [r for r in lines if r.get('pred_x') and r.get('gt_x')]
print(f'Records with both pred and GT: {len(sample)}')
if sample:
    print('Sample:', sample[0]['pred_x'], sample[0]['gt_x'])
"

Records with both pred and GT: 85
Sample: 0.467 0.47760416666666666
'''