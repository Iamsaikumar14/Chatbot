import sqlite3
import os
from dotenv import load_dotenv
import uuid


def generate_thread_id():
    return str(uuid.uuid4())

load_dotenv()
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

class ChatState(TypedDict):
    # Annotated list with add_messages automatically appends new messages
    messages: Annotated[list[BaseMessage], add_messages]


from langchain_core.tools import tool
from rag_service import RAGService

# Instantiate RAG service
rag_service = RAGService()

@tool
def search_documents(query: str) -> str:
    """Search uploaded documents/files for information matching the query. Use this tool when you need to answer questions using context from uploaded files/documents."""
    results = rag_service.query_documents(query, limit=4)
    if not results:
        return "No relevant information found in the uploaded documents."
    
    # Store retrieved sources in streamlit session state for frontend rendering (if running in streamlit context)
    try:
        import streamlit as st
        if "retrieved_sources" not in st.session_state:
            st.session_state.retrieved_sources = []
        for res in results:
            # Check if this source is already present to prevent duplicate listings
            exists = any(
                src["filename"] == res["filename"] and src["content"] == res["content"] 
                for src in st.session_state.retrieved_sources
            )
            if not exists:
                st.session_state.retrieved_sources.append({
                    "filename": res["filename"],
                    "content": res["content"],
                    "image_data": res.get("image_data"),
                    "similarity": res["similarity"]
                })
    except Exception:
        pass

    formatted_results = []
    for i, res in enumerate(results):
        formatted_results.append(
            f"Document: {res['filename']} (Similarity: {res['similarity']:.4f})\nContent: {res['content']}"
        )
    return "\n\n---\n\n".join(formatted_results)

# Define tools
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, search_documents]

# Initialize the Google Gemini model and bind tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatnode(state: ChatState):
    # Call LLM with the list of previous messages and tools
    response = llm_with_tools.invoke(state["messages"])
    # Return the response to update graph state
    return {"messages": [response]}

builder = StateGraph(ChatState)

# Add the nodes
builder.add_node("chatnode", chatnode)
builder.add_node("tools", ToolNode(tools))

# Define routing
builder.add_edge(START, "chatnode")
builder.add_conditional_edges(
    "chatnode",
    tools_condition,
)
builder.add_edge("tools", "chatnode")

# Add an checkpointer for conversation memory
connection = sqlite3.connect(database = 'chatbot.db' ,  check_same_thread=False)

memory = SqliteSaver(conn = connection)



# Compile the workflow
chatbot = builder.compile(checkpointer=memory)


# Conversation config with thread identifier
thread_id = "session-1"
config = {"configurable": {"thread_id": thread_id}}

def retrieve_all_threads():
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []

def delete_thread(thread_id_val):
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id_val,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id_val,))
        connection.commit()
    except sqlite3.OperationalError:
        pass

if __name__ == "__main__":
    # Test streaming
    print("Testing streaming:")
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content='what is the recipe to make pasta')]},
        config=config,
        stream_mode='messages'
    ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print("\n")





