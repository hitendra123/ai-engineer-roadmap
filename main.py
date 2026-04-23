from dotenv import load_dotenv
load_dotenv()

from graph import build_graph

def main():
    app = build_graph()
    session_id = "user-session-001"
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 50)
    print("AI Support Chatbot (LangGraph)")
    print("Type 'quit' to exit | 'new' to start fresh session")
    print("=" * 50)

    session_counter = 0
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            session_counter += 1
            config = {"configurable": {"thread_id": f"user-session-{session_counter:03d}"}}
            print("--- New session started ---")
            continue

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        )

        last_msg = result["messages"][-1].content
        intent = result.get("intent", "general")
        turn = result.get("turn_count", 0)

        print(f"\nBot [{intent} | turn {turn}]: {last_msg}")

        if result.get("summary"):
            print(f"\n[Session Summary]: {result['summary']}")

if __name__ == "__main__":
    main()