from bert_score import score as calculate_bert_score
from rag.generator import results

def bert_score(cands, refs):
    reference = ''.join(refs)
    score = calculate_bert_score(cands=[cands], refs=[reference], lang='en')
    return round(score[2].item(), 4)

def generate_bert_scores(results):
    print('BERT score initiated...')
    bert_scores = results.copy() # copy list to keep them distinct
    for result in bert_scores:
        result['BERT_score'] = bert_score(result['answer'], result['context'])
    print('Scores have been assigned. Retuning scores to pipeline.')
    return bert_scores

# for result in bert_scores:
#     print(f"Q: {result['question'][:50]}")
#     print(f"BERT: {result['BERT_score']}")
#     print("─" * 50)