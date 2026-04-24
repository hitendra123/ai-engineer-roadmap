import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()

FAISS_PATH = "faiss_db"


def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_query"
    )
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@traceable(name="RAG Query", tags=["rag", "week3"])
def ask(question: str, chain) -> dict:
    retriever = load_retriever()
    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "preview": doc.page_content[:150] + "..."
            }
            for doc in retrieved_docs
        ]
    }
