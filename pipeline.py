import json
import pandas as pd
from rag.generator import generate_results
from eval.BERT_score import generate_bert_scores
from eval.judge import judge

print('Pipeline initiated...')

results = generate_results()
scored = generate_bert_scores(results)
final = judge(scored)

print('Preparing data output...')

for entry in final:
    scores = json.loads(entry['raw_scores'])
    entry.update(scores)
    del entry['raw_scores'] 

print('Printing verdict...')

df = pd.DataFrame(final)
df.to_csv('results.csv', index=False)
print("Saved to results.csv")