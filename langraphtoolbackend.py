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
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, RemoveMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

class ChatState(TypedDict):
    # Annotated list with add_messages automatically appends new messages
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


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
@tool
def save_user_fact(fact: str) -> str:
    """Save a durable fact about the user (e.g. their field of study, role, or preferences) to long-term memory.
    Use this tool when the user tells you something personal or important about themselves that should be remembered across chat sessions."""
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO long_term_memory (fact) VALUES (?)", (fact.strip(),))
        conn.commit()
        return f"Successfully saved fact to long-term memory: '{fact}'"
    except sqlite3.IntegrityError:
        return f"This fact is already saved in long-term memory: '{fact}'"
    except Exception as e:
        return f"Error saving fact: {str(e)}"
    finally:
        conn.close()

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information using DuckDuckGo. Use this tool when you need information that is not available in the local uploaded documents."""
    import time
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    
    # Try up to 3 times with backoff
    for attempt in range(3):
        try:
            result = search.run(query)
            if result and not "Error" in result:
                return result
        except Exception as e:
            print(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1.0)
                
    return "Search service is temporarily unavailable. Using internal cached knowledge to answer."

search_tool = search_web
tools = [search_web, search_documents, list_calendar_events,
    create_calendar_event,
    delete_calendar_event,
    update_calendar_event,
    save_user_fact]

# Initialize the Google Gemini model and bind tools
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatnode(state: ChatState):
    # Prepend system prompt containing current datetime in local time (IST)
    current_time_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    system_prompt = f"""You are a helpful AI Assistant for NIT Rourkela.
The current date and time is: {current_time_str}. The timezone is Indian Standard Time (IST, UTC+05:30).
You have tools to search documents, search the web, manage the user's Google Calendar, and save durable user facts.
When the user asks you to manage their calendar (list, create, update, or delete events), call the corresponding calendar tool.
If a calendar operation fails because it is not connected, politely guide the user to connect it using the settings panel in the sidebar.
Always format start and end times in ISO 8601 format with the appropriate local timezone offset (+05:30).
"""
    # 1. Fetch long-term memory user facts
    memories = []
    try:
        conn = sqlite3.connect('chatbot.db')
        cursor = conn.cursor()
        cursor.execute("SELECT fact FROM long_term_memory ORDER BY created_at DESC")
        memories = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception:
        pass
        
    if memories:
        memory_text = "\n".join(f"- {m}" for m in memories)
        system_prompt += f"\n\nHere are durable facts about the user from long-term memory (personalization context):\n{memory_text}"
        
    # 2. Append existing summary if present
    summary = state.get("summary", "")
    if summary:
        system_prompt += f"\n\nHere is a summary of the earlier conversation history:\n{summary}"

    # Retrieve last user message to inject context automatically
    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
            
    if last_user_message:
        # Search for highly relevant chunks (similarity >= 0.65)
        rag_results = rag_service.query_documents(last_user_message, limit=3)
        relevant_results = [r for r in rag_results if r["similarity"] >= 0.65]
        if relevant_results:
            context_blocks = []
            for r in relevant_results:
                context_blocks.append(f"Source Document: {r['filename']} (Similarity: {r['similarity']:.4f})\nContent:\n{r['content']}")
            rag_context = "\n\n---\n\n".join(context_blocks)
            system_prompt += f"\n\nHere is highly relevant context retrieved from the user's uploaded documents:\n{rag_context}\nUse this context to help answer the user's question directly without calling the search tool."
            
            # Store retrieved sources in streamlit session state for frontend rendering
            try:
                import streamlit as st
                if "retrieved_sources" not in st.session_state:
                    st.session_state.retrieved_sources = []
                for res in relevant_results:
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

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # 3. Check for conversation summarization (if total message count exceeds 10)
    current_messages = state["messages"] + [response]
    new_summary = summary
    remove_messages = []
    
    if len(current_messages) > 10:
        # Keep only the last 4 messages in state, summarize all preceding messages
        to_summarize = current_messages[:-4]
        formatted_history = ""
        for m in to_summarize:
            role = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else "Tool"
            content_str = m.content if isinstance(m.content, str) else str(m.content)
            formatted_history += f"{role}: {content_str}\n"
            
        summary_prompt = f"""Summarize the following conversation history concisely, focusing on key facts and user preferences discussed.
Existing Summary: {summary}

New History:
{formatted_history}

Concise Summary:"""
        try:
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            new_summary = summary_response.content.strip()
            # Generate RemoveMessage instructions for the database checkpointer
            remove_messages = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
        except Exception as e:
            print(f"Error during summarization: {e}")
            
    return {
        "messages": [response] + remove_messages,
        "summary": new_summary
    }


def grader_node(state: ChatState):
    """
    Grader node (CRAG) that runs after tools. If 'search_documents' was run and
    the results are graded as insufficient by a lightweight LLM check, we automatically
    fall back to web search and replace/amend the ToolMessage.
    """
    print("\n--- GRADER NODE CALLED ---")
    messages = state["messages"]
    if not messages:
        print("Grader: No messages found in state.")
        return {"messages": []}
        
    last_message = messages[-1]
    print(f"Grader: Last message class = {last_message.__class__.__name__}, name = {getattr(last_message, 'name', None)}")
    
    # Check if the last message is a ToolMessage from search_documents
    if isinstance(last_message, ToolMessage) and last_message.name == "search_documents":
        # Get the original user question
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
                
        print(f"Grader: Evaluating search_documents result for query: '{user_query}'")
        if user_query:
            # Grader prompt for lightweight LLM evaluation
            grader_prompt = f"""You are an objective QA Grader.
Evaluate whether the retrieved document chunks contain sufficient information to answer the user's question.
If the documents contain enough information to answer the question, respond with exactly 'YES'.
If the documents are irrelevant, empty, insufficient, or do not contain the answer, respond with exactly 'NO'.

User Question: {user_query}
Retrieved Documents:
{last_message.content}

Response (YES/NO):"""
            try:
                # Lightweight model call
                response = llm.invoke([HumanMessage(content=grader_prompt)])
                grade = response.content.strip().upper()
                print(f"Grader: LLM response grade = '{grade}'")
            except Exception as e:
                print(f"Grader: Error during grading: {e}")
                grade = "YES" # Default to YES on API error to avoid blocking
                
            if "NO" in grade:
                print("Grader: Documents are INSUFFICIENT. Falling back to Web Search...")
                # Fall back to web search (DuckDuckGo)
                try:
                    web_results = search_tool.invoke(user_query)
                    print("Grader: Web Search returned results successfully.")
                except Exception as e:
                    web_results = f"Web search fallback failed: {str(e)}"
                    print(f"Grader: Web Search failed with error: {e}")
                    
                corrective_content = f"""[Corrective RAG: Local documents were graded as INSUFFICIENT. Falling back to Web Search...]

Web Search Results:
{web_results}"""
                
                # Replace the last tool message content with the web search fallback content by matching its ID
                fallback_message = ToolMessage(
                    content=corrective_content,
                    name=last_message.name,
                    tool_call_id=last_message.tool_call_id,
                    id=last_message.id
                )
                print("Grader: Returning fallback ToolMessage replacement.")
                return {"messages": [fallback_message]}
            else:
                print("Grader: Documents are SUFFICIENT. Proceeding to generation.")
                
    return {"messages": []}


builder = StateGraph(ChatState)

# Add the nodes
builder.add_node("chatnode", chatnode)
builder.add_node("tools", ToolNode(tools))
builder.add_node("grader", grader_node)

# Define routing
builder.add_edge(START, "chatnode")
builder.add_conditional_edges(
    "chatnode",
    tools_condition,
)
builder.add_edge("tools", "grader")
builder.add_edge("grader", "chatnode")

# Add an checkpointer for conversation memory and create long-term memory table
connection = sqlite3.connect(database = 'chatbot.db' ,  check_same_thread=False)
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS long_term_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
connection.commit()

memory = SqliteSaver(conn = connection)



# Compile the workflow with human-in-the-loop interruption before tools are run
chatbot = builder.compile(checkpointer=memory, interrupt_before=["tools"])


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





