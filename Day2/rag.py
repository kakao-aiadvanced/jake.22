import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# set up
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your Langchain API key: ")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-granular-depot-30"

# llm
llm = ChatOpenAI(model="gpt-4o-mini")

# load
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
loader = WebBaseLoader(urls)
docs = loader.load()

# split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splits = text_splitter.split_documents(docs)

# embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# relevance checker
json_parser = JsonOutputParser()
relevance_checker_prompt = """
## Document Relevance Assessment

**Instructions:**
Follow these rules strictly:
1. Assess the retrieved chunk based only on its direct relevance to the user's query.
2. If the chunk provides information that is useful or closely related to the query, classify it as relevant.
3. If the chunk is off-topic, irrelevant, or only tangentially related, classify it as not relevant.
4. Your output must be in JSON format with a single key-value pair: 
   - If relevant, return {{ "relevance": "yes" }}.
   - If not relevant, return {{ "relevance": "no" }}.

**User Query:** {query}

**Retrieved Documents:**
{retrieved_docs}

**Output Format:**
Provide a JSON list of dictionaries, where each dictionary corresponds to a document and contains a 'relevance' assessment.

Example:
[
    {{'relevance': 'yes'}},  # If the first document is relevant
    {{'relevance': 'no'}},   # If the second document is not relevant
    ...
]

**Assessment:**
"""
relevance_checker_prompt_template = PromptTemplate(
    template=relevance_checker_prompt,
    input_variables=["query", "retrieved_docs"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)
relevance_checker = relevance_checker_prompt_template | llm | json_parser
def check_relevance(question):
    retrieved_docs = retriever.invoke(question)
    return retrieved_docs, relevance_checker.invoke({"query": question, "retrieved_docs": [context.page_content for context in retrieved_docs]})

# print(check_relevance("llm agent에 대해 설명해줘")[1])
# print(check_relevance("한국 음식 추천해줘")[1])


# generate answer
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

answer_prompt = PromptTemplate(
    template="""
    **Context:**
    {context}

    **User Question:** {question}

    **Instructions:**
    Refer to the context to answer the user's question.

    **Answer:**
    """,
    input_variables=["context, question"],
)



def query(question, retrieved_docs, relevance_results):
    if all(map(lambda x: x["relevance"] == "yes", relevance_results)):
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | answer_prompt
                | llm
                | StrOutputParser()
        )
        return "".join(rag_chain.stream(question))

    else:
        return "Sorry, I am unable to find the relevant context to answer."

# question = "llm agent에 대해 설명해줘"
# retrieved_docs, relevance_results = check_relevance(question)
# query(question, retrieved_docs, relevance_results)


# hallucination checker
hallucination_prompt = PromptTemplate(
    template="""
    ## 주어진 컨텍스트 내에서만 답변하고, 컨텍스트에서 찾을 수 없는 정보를 사용하여 답변을 생성하지 마세요.

**질문:** {question}

**컨텍스트:** {context}

**대답:** {generated_answer}

**할루시네이션 체크:**

1. 대답이 주어진 컨텍스트 내의 정보만을 사용하여 생성되었습니까?
2. 대답에 컨텍스트에서 찾을 수 없는 정보나 사실이 포함되어 있습니까?
3. 대답에 모호하거나 불확실한 내용이 포함되어 있습니까?
4. 대답이 질문과 관련이 있고 질문에 대한 답변을 제공합니까?
5. 대답이 논리적이고 일관성이 있습니까?

**평가:**

* **할루시네이션 없음:** 대답이 컨텍스트 내의 정보만을 사용하여 생성되었고, 정확하고 신뢰할 수 있습니다. 출력: {{"hallucination": "no"}}
* **할루시네이션 있음:** 대답에 컨텍스트에서 찾을 수 없는 정보가 포함되어 있거나, 모호하거나 불확실하거나, 질문과 관련이 없거나, 논리적이지 않거나 일관성이 없습니다. 출력: {{"hallucination": "yes"}}

**추가 정보:**

* 할루시네이션은 대규모 언어 모델에서 흔히 발생하는 문제입니다.
* 할루시네이션을 줄이기 위해서는 컨텍스트를 명확하게 제공하고, 대답의 출처를 확인하는 것이 중요합니다.
* 할루시네이션이 의심되는 경우, 추가적인 정보를 찾아 확인하는 것이 좋습니다.
""",
    input_variables=["generated_answer", "context", "question"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

def check_hallucination(question, generated_answer, retrieved_docs):
    chain = hallucination_prompt | llm | json_parser
    return chain.invoke({"generated_answer": generated_answer, "context": format_docs(retrieved_docs), "question": question})


# question = "llm agent에 대해 설명해줘"
# retrieved_docs, relevance_results = check_relevance(question)
# generated_answer = query(question, retrieved_docs, relevance_results)
# print(generated_answer)
# print(check_hallucination(question, generated_answer, retrieved_docs))


# retry
def query_with_retry(question, retry = True):
    retrieved_docs, relevance_results = check_relevance(question)
    generated_answer = query(question, retrieved_docs, relevance_results)
    if retry and check_hallucination(question, generated_answer, retrieved_docs)["hallucination"] == "yes":
        print("retry")
        return query_with_retry(question, False)
    return generated_answer

print(query_with_retry("llm agent에 대해 설명해줘"))
print(query_with_retry("한국 음식에 대해 설명해줘"))
