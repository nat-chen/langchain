import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from fastapi import FastAPI
from langserve import add_routes
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = 'xxx'
os.environ['OPENAI_API_BASE'] = 'xxx'

# loader = WebBaseLoader(
#     web_path="https://www.gov.cn/jrzg/2013-10/25/content_2515601.htm",
#     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#             class_=("p1")
#         ))
# )
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# db = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
prompt_template_str = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
prompt_template = PromptTemplate.from_template(prompt_template_str)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
app = FastAPI(
  title="消费者权益智能助手",
  version="1.0",
)
# 3. Adding chain route
add_routes(
    app,
    rag_chain,
    path="/consumer_ai",
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
