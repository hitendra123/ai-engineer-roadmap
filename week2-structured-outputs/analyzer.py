import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from models import JobAnalysis

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

structured_llm = llm.with_structured_output(JobAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert job analyzer and career advisor.
Extract structured information from job descriptions accurately.
For match_score, evaluate against this profile:
- Strong: Python, LangChain, RAG, Azure AI, LangGraph, Vector Databases
- Moderate: JavaScript, SQL, Docker
- Weak: Kubernetes, Java, mobile development"""),
    ("human", "Analyze this job posting and extract all details:\n\n{job_description}")
])

chain = prompt | structured_llm


def analyze_job(job_description: str) -> JobAnalysis:
    return chain.invoke({"job_description": job_description})


def analyze_multiple(job_descriptions: list[str]) -> list[JobAnalysis]:
    results = []
    for i, job in enumerate(job_descriptions, 1):
        print(f"Analyzing job {i}/{len(job_descriptions)}...")
        result = analyze_job(job)
        results.append(result)
    return sorted(results, key=lambda x: x.match_score, reverse=True)
