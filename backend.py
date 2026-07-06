import os
from dotenv import load_dotenv

load_dotenv()
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    # Annotated list with add_messages automatically appends new messages
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chatnode(state: ChatState):
    # Call LLM with the list of previous messages
    response = llm.invoke(state["messages"])
    # Return the response to update graph state
    return {"messages": [response]}

builder = StateGraph(ChatState)

# Add the chatnode node
builder.add_node("chatnode", chatnode)

# Define routing
builder.add_edge(START, "chatnode")
builder.add_edge("chatnode", END)

# Add an in-memory checkpointer for conversation memory
memory = MemorySaver()

# Compile the workflow
chatbot = builder.compile(checkpointer=memory)

# Conversation config with thread identifier
config = {"configurable": {"thread_id": "session-1"}}

if __name__ == "__main__":
    # First query
    initial_state = {
        "messages": [HumanMessage(content="what is capital of India")]
    }

    result = chatbot.invoke(initial_state, config)

    # Print response
    print("AI:", result["messages"][-1].content)

