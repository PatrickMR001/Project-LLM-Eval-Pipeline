import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('ANTHROPIC_API_KEY')

client = anthropic.Anthropic(api_key=api_key)

system_prompt = '''You are an evaluation judge. Rate the generated answer compared to the question and context on a scale from 0 to 1.
You are given the inital question, the to be checked answer and the given context for the first answer. Return following:
1. Your judgment as a single number from 0 to 1. You are also given a BERTScore which measures semantic similarity between the answer and context (0-1, higher is better). Use it as additional signal for your judgment. For your judgemet follow this rule: 1 being the best outcome, 0 the worst and up to four decimal points.
2. The faithfulness: Given context and answer, is the answer fully supported by the context? Return a score from 0-1 with 1 being the best outcome, 0 the worst and up to four decimal points.
3. The answer relevancy: Does the answer actually address the question? Return a score from 0-1 with 1 being the best outcome, 0 the worst and up to four decimal points.
4. The context recall: Given the question, was the right context retrieved? Return a score from 0-1 with 1 being the best outcome, 0 the worst and up to four decimal points.
Return valid JSON with double quotes only, scores as numbers not strings:
{"LLM_judgment": 0.85, "Faithfulness": 0.90, "Answer_Relevancy": 0.75, "Context_Recall": 0.80}'''

def api_call(results):
    response = client.messages.create(
        model='claude-opus-4-5',
        max_tokens=150,
        system=system_prompt,
        messages=[
            {'role': 'user', 'content':f"question: {results['question']}\n\nanswer: {results['answer']}\n\ncontext: {results['context']}\n\nBERTScore: {results['BERT_score']}"}
        ]
    )
    return response

def judge(bert_scores):
    print('Judge initiated...')
    for entry in bert_scores:
        entry['raw_scores'] = api_call(entry).content[0].text
    print('All done here. Returning data to pipeline.')
    return bert_scores