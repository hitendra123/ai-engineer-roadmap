import os
from langchain_google_genai import ChatGoogleGenerativeAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import SystemMessage, HumanMessage
from state import SupportState

llm = ChatGoogleGenerativeAI(
      model="models/gemini-2.5-flash",
      temperature=0.7,
      google_api_key=os.getenv("GOOGLE_API_KEY")
  )

def classify_intent(state: SupportState) -> SupportState:
      last_msg = state["messages"][-1].content.lower()
      if any(w in last_msg for w in ["code", "error", "api", "debug", "install", "python"]):
          intent = "technical"
      elif any(w in last_msg for w in ["price", "billing", "cost", "refund", "payment"]):
          intent = "billing"
      else:
          intent = "general"
      return {"intent": intent, "turn_count": state.get("turn_count", 0) + 1}

def technical_node(state: SupportState) -> SupportState:
    system = SystemMessage(content="You are a senior AI engineer. Be precise, include code examples.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def billing_node(state: SupportState) -> SupportState:
    system = SystemMessage(content="You are a billing specialist. Be empathetic and solution-focused.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def general_node(state: SupportState) -> SupportState:
    system = SystemMessage(content="You are a friendly assistant. Keep answers simple and concise.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def summarize_node(state: SupportState) -> SupportState:
    if state.get("turn_count", 0) % 10 == 0 and len(state["messages"]) > 10:
        history = "\n".join([m.content for m in state["messages"][-10:]])
        summary = llm.invoke([HumanMessage(content=f"Summarize in 2 sentences:\n{history}")])
        return {"summary": summary.content}
    return {}