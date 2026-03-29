from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from data.questions import questions


load_dotenv()
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory='/Users/patrickmischkeramirez/Documents/Job Practice /Portfolio/backend/chroma_db',
    embedding_function=embeddings
)

results = []

# LLM
llm = ChatOpenAI(model='gpt-4o-mini', max_completion_tokens=2048)

# RAG
retriever = vectorstore.as_retriever(search_kwargs={'k':4})

# prompt
system_prompt = '''You help recruiters understand why Patrick is a great candidate. 
Answer only based on this context: {context}
You can make reasonable inferences from the information provided.
If the answer is truly not in the context, say "I do not have this information".'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

output_parser = StrOutputParser()

# Build chain 
chain = (
    {
        'context': itemgetter('input') | retriever | format_docs,
        'input': itemgetter('input')
    }   
    | prompt 
    | llm 
    | output_parser
)

def generate_results():
    print('Initiating generator...')
    results = []
    for question in questions:
        docs = retriever.invoke(question['question'])
        context = [doc.page_content for doc in docs]

        response = chain.invoke(
            {'input': question['question']}
        )

        results.append({
            'question': question['question'],
            'answer': response,
            'context': context,
        })
    print('All done. Returning results back to pipeline...')
    return results

