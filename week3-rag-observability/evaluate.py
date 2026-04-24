import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from rag_chain import load_retriever, build_chain

load_dotenv()

TEST_QUESTIONS_FILE = "test_questions.json"


def load_test_questions():
    if not os.path.exists(TEST_QUESTIONS_FILE):
        print(f"No {TEST_QUESTIONS_FILE} found. Creating sample file...")
        sample = [
            {"input": "What is the main topic of the document?"},
            {"input": "What are the key conclusions?"},
            {"input": "Who are the main people or organizations mentioned?"},
        ]
        with open(TEST_QUESTIONS_FILE, "w") as f:
            json.dump(sample, f, indent=2)
        print(f"Created {TEST_QUESTIONS_FILE} — edit it with real questions about your document")
        return sample
    with open(TEST_QUESTIONS_FILE) as f:
        return json.load(f)


def run_evaluation():
    print("Loading RAG pipeline...")
    retriever = load_retriever()
    chain = build_chain(retriever)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    questions = load_test_questions()
    print(f"Running evaluation on {len(questions)} questions...")

    def rag_pipeline(inputs: dict) -> dict:
        answer = chain.invoke(inputs["input"])
        return {"output": answer}

    client = Client()

    dataset_name = "week3-rag-test-dataset"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name)
        client.create_examples(
            inputs=[{"input": q["input"]} for q in questions],
            dataset_id=dataset.id
        )
        print(f"Created dataset '{dataset_name}' in LangSmith")

    evaluators = [
        LangChainStringEvaluator("criteria", config={
            "criteria": "helpfulness",
            "llm": llm
        }),
        LangChainStringEvaluator("criteria", config={
            "criteria": "conciseness",
            "llm": llm
        }),
    ]

    results = evaluate(
        rag_pipeline,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="week3-eval",
        metadata={"model": "gemini-1.5-flash", "retriever": "chromadb-k3"}
    )

    print("\nEvaluation complete! View results at smith.langchain.com")
    print(f"Experiment: week3-eval")
    return results


if __name__ == "__main__":
    run_evaluation()
