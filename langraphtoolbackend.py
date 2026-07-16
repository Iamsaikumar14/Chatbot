import sqlite3
import os
from dotenv import load_dotenv
import uuid
from langchain_core.messages import SystemMessage
from datetime import datetime



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
import calendar_service

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
@tool
def list_calendar_events(max_results: int = 10, time_min: str = None, time_max: str = None) -> str:
    """List upcoming events from the user's Google Calendar.
    - time_min: ISO 8601 string (e.g., '2026-07-16T00:00:00+05:30'). Defaults to current time if not provided.
    - time_max: Optional ISO 8601 string to restrict the end time.
    """
    if not calendar_service.is_connected():
        return "Error: Google Calendar is not connected. Suggest that the user connect it in the sidebar settings."
    
    events = calendar_service.list_events(max_results=max_results, time_min=time_min, time_max=time_max)
    if isinstance(events, str):
        return events
    if not events:
        return "No upcoming events found."
        
    lines = []
    for ev in events:
        start = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
        end = ev.get('end', {}).get('dateTime') or ev.get('end', {}).get('date')
        lines.append(f"- **{ev.get('summary')}**\n  ID: {ev.get('id')}\n  Start: {start}\n  End: {end}\n  Description: {ev.get('description', 'N/A')}")
    return "\n".join(lines)

@tool
def create_calendar_event(summary: str, start_time: str, end_time: str, description: str = None, location: str = None) -> str:
    """Create a new event in the Google Calendar.
    - summary: Title of the event.
    - start_time: ISO 8601 formatted datetime string (e.g. '2026-07-16T15:00:00+05:30').
    - end_time: ISO 8601 formatted datetime string.
    - description: Optional details.
    - location: Optional location.
    """
    if not calendar_service.is_connected():
        return "Error: Google Calendar is not connected. Suggest that the user connect it in the sidebar settings."
    return calendar_service.create_event(summary, start_time, end_time, description, location)

@tool
def delete_calendar_event(event_id: str) -> str:
    """Delete an event from Google Calendar using its event_id."""
    if not calendar_service.is_connected():
        return "Error: Google Calendar is not connected. Suggest that the user connect it in the sidebar."
    return calendar_service.delete_event(event_id)

@tool
def update_calendar_event(event_id: str, summary: str = None, start_time: str = None, end_time: str = None, description: str = None, location: str = None) -> str:
    """Update details of an existing event in Google Calendar.
    Provide only the parameters that need to be modified.
    """
    if not calendar_service.is_connected():
        return "Error: Google Calendar is not connected. Suggest that the user connect it in the sidebar."
    return calendar_service.update_event(event_id, summary, start_time, end_time, description, location)
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, search_documents, list_calendar_events,
    create_calendar_event,
    delete_calendar_event,
    update_calendar_event]

# Initialize the Google Gemini model and bind tools
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatnode(state: ChatState):
    # Prepend system prompt containing current datetime in local time (IST)
    current_time_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    system_prompt = f"""You are a helpful AI Assistant for NIT Rourkela.
The current date and time is: {current_time_str}. The timezone is Indian Standard Time (IST, UTC+05:30).
You have tools to search documents, search the web, and manage the user's Google Calendar.
When the user asks you to manage their calendar (list, create, update, or delete events), call the corresponding calendar tool.
If a calendar operation fails because it is not connected, politely guide the user to connect it using the settings panel in the sidebar.
Always format start and end times in ISO 8601 format with the appropriate local timezone offset (+05:30).
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
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





