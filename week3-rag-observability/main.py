import os
from dotenv import load_dotenv
from rag_chain import load_retriever, build_chain, ask

load_dotenv()


def check_setup():
    if not os.path.exists("chroma_db"):
        print("No vector store found. Run ingest.py first:")
        print("  python ingest.py")
        return False
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Warning: LANGCHAIN_API_KEY not set — tracing disabled")
    else:
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print(f"LangSmith tracing enabled — project: {project}")
        print(f"View traces at: https://smith.langchain.com")
    return True


def main():
    print("\nRAG App with LangSmith Observability")
    print("=====================================")

    if not check_setup():
        return

    print("Loading RAG pipeline...")
    retriever = load_retriever()
    chain = build_chain(retriever)
    print("Ready. Every query is traced in LangSmith.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break

        print("\nSearching documents...")
        result = ask(question, chain)

        print(f"\nAnswer:\n{result['answer']}")

        print(f"\nSources used ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src['source']} | Page {src['page']}")
            print(f"     \"{src['preview']}\"")

        print(f"\n[Trace logged to LangSmith — smith.langchain.com]")
        print("-" * 50)


if __name__ == "__main__":
    main()
