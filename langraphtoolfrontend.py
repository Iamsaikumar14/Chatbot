import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langraphtoolbackend import chatbot, generate_thread_id, retrieve_all_threads, delete_thread
import pypdf
from rag_service import RAGService
import calendar_service

# Instantiate RAG service
rag_service = RAGService()

def extract_text_from_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                elif "content" in part:
                    parts.append(part["content"])
        return "".join(parts)
    return str(content)

# Set page title and layout
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)



# Custom premium CSS styling for the chat interface
st.markdown("""
<style>
    /* Styling for the main app container */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f1f5f9;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    }
    
    /* Center header title and add modern typography */
    h1 {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 5px !important;
        font-size: 2.8rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Style the chat input bar */
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Application header
st.markdown("<h1>Gemini & LangGraph Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A stateful conversational AI with streaming responses</p>", unsafe_allow_html=True)

# Initialize session state variables
if "threads" not in st.session_state:
    try:
        existing_thread_ids = retrieve_all_threads()
    except Exception:
        existing_thread_ids = []
    
    st.session_state.threads = []
    for tid in existing_thread_ids:
        try:
            state = chatbot.get_state({"configurable": {"thread_id": tid}})
            messages = state.values.get("messages", [])
            title = "New Chat"
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    clean_content = extract_text_from_content(msg.content)
                    title = clean_content[:25] + "..." if len(clean_content) > 25 else clean_content
                    break
        except Exception:
            title = "New Chat"
        st.session_state.threads.append({"id": tid, "title": title})

if "thread_id" not in st.session_state:
    if st.session_state.threads:
        st.session_state.thread_id = st.session_state.threads[0]["id"]
    else:
        new_id = generate_thread_id()
        st.session_state.thread_id = new_id

if "last_retrieved_sources" not in st.session_state:
    st.session_state.last_retrieved_sources = []

# Ensure active thread_id is in the list of threads and at the top
active_thread = next((t for t in st.session_state.threads if t["id"] == st.session_state.thread_id), None)
if active_thread:
    # Move active thread to the top (index 0)
    st.session_state.threads = [t for t in st.session_state.threads if t["id"] != st.session_state.thread_id]
    st.session_state.threads.insert(0, active_thread)
else:
    # Create and insert at the top
    st.session_state.threads.insert(0, {"id": st.session_state.thread_id, "title": "New Chat"})

# Set up active thread configuration
active_config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "metadata" : {
        "thread_id" : st.session_state.thread_id
    },
    "run_name" : "chat_turn"
}

# Sidebar UI
st.sidebar.title('LangGraph Chatbot')

# "New Chat" button
if st.sidebar.button('➕ New Chat', use_container_width=True):
    new_id = generate_thread_id()
    st.session_state.thread_id = new_id
    st.session_state.threads.insert(0, {"id": new_id, "title": "New Chat"})
    st.session_state.last_retrieved_sources = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader('My Conversations')

# Render conversations in sidebar as buttons with delete option
for thread in st.session_state.threads:
    # Use columns to align the chat selection button and delete button
    col1, col2 = st.sidebar.columns([5, 1])
    
    # Highlight current active thread
    is_active = (thread["id"] == st.session_state.thread_id)
    button_label = f"💬 {thread['title']}"
    if is_active:
        button_label = f"👉 {thread['title']} (Active)"
    
    # Click to switch thread
    if col1.button(button_label, key=f"select_{thread['id']}", use_container_width=True):
        st.session_state.thread_id = thread["id"]
        st.session_state.last_retrieved_sources = []
        st.rerun()
        
    # Click to delete thread
    if col2.button("🗑️", key=f"delete_{thread['id']}", use_container_width=True, help="Delete this conversation"):
        # Delete thread checkpoints from database
        delete_thread(thread["id"])
        
        # Filter out the deleted thread
        st.session_state.threads = [t for t in st.session_state.threads if t["id"] != thread["id"]]
        
        # If the deleted thread was the active one, choose a new active thread
        if st.session_state.thread_id == thread["id"]:
            if st.session_state.threads:
                st.session_state.thread_id = st.session_state.threads[0]["id"]
            else:
                new_id = generate_thread_id()
                st.session_state.thread_id = new_id
                st.session_state.threads.insert(0, {"id": new_id, "title": "New Chat"})
        
        st.rerun()

# Document Center (RAG) in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Document Center (RAG)")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload docs or images (png, jpg, jpeg)",
    type=["txt", "md", "pdf", "csv", "json", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="rag_uploader"
)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set([d["filename"] for d in rag_service.list_documents()])
            
        if filename not in st.session_state.processed_files:
            with st.sidebar.status(f"Processing {filename}...", expanded=True) as status:
                try:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                         import base64
                         file_bytes = uploaded_file.read()
                         base64_str = base64.b64encode(file_bytes).decode("utf-8")
                         content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                         
                         rag_service.add_image(filename, base64_str, content_type)
                         st.session_state.processed_files.add(filename)
                         status.update(label=f"✅ Indexed Image {filename}!", state="complete")
                    elif filename.endswith(".pdf"):
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        
                        if text.strip():
                            rag_service.add_document(filename, text)
                            st.session_state.processed_files.add(filename)
                            status.update(label=f"✅ Indexed {filename}!", state="complete")
                        else:
                            status.update(label=f"⚠️ {filename} is empty!", state="error")
                    else:
                        text = uploaded_file.read().decode("utf-8", errors="ignore")
                        if text.strip():
                            rag_service.add_document(filename, text)
                            st.session_state.processed_files.add(filename)
                            status.update(label=f"✅ Indexed {filename}!", state="complete")
                        else:
                            status.update(label=f"⚠️ {filename} is empty!", state="error")
                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)}", state="error")

# List existing documents in the database
indexed_docs = rag_service.list_documents()
if indexed_docs:
    st.sidebar.markdown("**Indexed Documents:**")
    for doc in indexed_docs:
        col_name, col_del = st.sidebar.columns([4, 1])
        col_name.write(f"📄 {doc['filename']}")
        if col_del.button("🗑️", key=f"del_doc_{doc['filename']}", help=f"Delete {doc['filename']}"):
            rag_service.delete_document(doc['filename'])
            if "processed_files" in st.session_state:
                st.session_state.processed_files.discard(doc['filename'])
            st.rerun()
else:
    st.sidebar.info("No documents uploaded yet.")
# --- Google Calendar Sidebar Integration ---
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Google Calendar")

if calendar_service.is_connected():
    st.sidebar.success("🟢 Connected")
    
    # Disconnect action
    if st.sidebar.button("🔌 Disconnect Calendar", use_container_width=True):
        calendar_service.disconnect()
        st.toast("Disconnected Google Calendar.")
        st.rerun()
        
    # Render upcoming events directly in sidebar
    st.sidebar.markdown("**Upcoming Events:**")
    events = calendar_service.list_events(max_results=5)
    if isinstance(events, list) and events:
        from datetime import datetime
        for ev in events:
            summary = ev.get('summary', 'Untitled Event')
            start = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
            try:
                dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                time_str = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                time_str = start
                
            st.sidebar.markdown(f"**{summary}**\n*{time_str}*")
            st.sidebar.markdown("---")
    elif isinstance(events, list):
        st.sidebar.info("No upcoming events found.")
    else:
        st.sidebar.error("Error loading events.")
else:
    st.sidebar.error("🔴 Disconnected")
    
    # Credentials setup expander
    with st.sidebar.expander("🔑 Setup Credentials", expanded=not calendar_service.has_client_credentials()):
        st.markdown(
            "To connect your Google Calendar, paste your OAuth Web Application Client Credentials below. "
            "Make sure your Redirect URI in the Google Cloud Console is set to `http://localhost:8501/`."
        )
        
        # Upload credentials.json or manually enter credentials
        uploaded_creds = st.file_uploader("Upload credentials.json", type=["json"], key="creds_uploader")
        if uploaded_creds:
            try:
                import json
                creds_data = json.load(uploaded_creds)
                with open(calendar_service.CREDENTIALS_PATH, 'w') as f:
                    json.dump(creds_data, f, indent=4)
                st.success("credentials.json saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                
        st.markdown("Or enter details manually:")
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
        
        if st.button("Save Credentials", use_container_width=True):
            if client_id and client_secret:
                calendar_service.save_client_credentials(client_id, client_secret)
                st.success("Credentials saved!")
                st.rerun()
            else:
                st.warning("Please fill in both fields.")
                
    # If credentials exist, show authorize button
    if st.sidebar.button("🔗 Connect Google Calendar", use_container_width=True):
        try:
            calendar_service.connect_google_calendar()
            st.success("Google Calendar Connected!")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# Handle Human-in-the-Loop Resume/Reject actions
if st.session_state.get("action") == "approve":
    st.session_state.action = None
    with st.chat_message("assistant"):
        try:
            # Resume stream by passing None as state update
            def resume_stream():
                for message_chunk, metadata in chatbot.stream(
                    None,
                    config=active_config,
                    stream_mode="messages"
                ):
                    if message_chunk.content:
                        clean_chunk = extract_text_from_content(message_chunk.content)
                        if clean_chunk:
                            yield clean_chunk
            st.write_stream(resume_stream())
            # Save retrieved sources if any
            if "retrieved_sources" in st.session_state:
                st.session_state.last_retrieved_sources = st.session_state.retrieved_sources
            st.rerun()
        except Exception as e:
            st.error(f"Error resuming chatbot: {e}")

elif st.session_state.get("action") == "reject":
    st.session_state.action = None
    state = chatbot.get_state(active_config)
    chat_history = state.values.get("messages", [])
    last_msg = chat_history[-1] if chat_history else None
    
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        reject_messages = []
        for tc in last_msg.tool_calls:
            reject_messages.append(ToolMessage(
                content="Permission denied by user. Do not attempt this action again.",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
        
        # Inject the rejected ToolMessages and specify we are writing as the 'tools' node output
        chatbot.update_state(active_config, {"messages": reject_messages}, as_node="tools")
        
        with st.chat_message("assistant"):
            try:
                # Resume execution with rejection injected
                def resume_stream_rejected():
                    for message_chunk, metadata in chatbot.stream(
                        None,
                        config=active_config,
                        stream_mode="messages"
                    ):
                        if message_chunk.content:
                            clean_chunk = extract_text_from_content(message_chunk.content)
                            if clean_chunk:
                                yield clean_chunk
                st.write_stream(resume_stream_rejected())
                st.rerun()
            except Exception as e:
                st.error(f"Error resuming chatbot: {e}")

# Display current message history for the active thread
state = chatbot.get_state(active_config)
chat_history = state.values.get("messages", [])

for idx, msg in enumerate(chat_history):
    clean_content = extract_text_from_content(msg.content)
    # Skip rendering empty tool calls or checkpointer state summaries to keep chat clean
    if not clean_content.strip() and not isinstance(msg, HumanMessage):
        continue
        
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(clean_content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(clean_content)
            
            # Show thumbs up/down feedback widget under the last assistant response
            if idx == len(chat_history) - 1:
                # We use the thread ID and message history length to construct a unique state key
                feedback = st.feedback("thumbs", key=f"fb_{st.session_state.thread_id}_{len(chat_history)}")
                if feedback is not None:
                    # Streamlit feedback returns 0 for thumbs down, 1 for thumbs up
                    val = 1 if feedback == 1 else -1
                    
                    # Locate the preceding human user question to log
                    last_query = ""
                    for m in reversed(chat_history[:idx]):
                        if isinstance(m, HumanMessage):
                            last_query = extract_text_from_content(m.content)
                            break
                            
                    if st.session_state.get("last_retrieved_sources"):
                        for src in st.session_state.last_retrieved_sources:
                            rag_service.add_feedback(
                                query=last_query,
                                filename=src["filename"],
                                chunk_content=src["content"],
                                feedback_value=val
                            )
                        if val == 1:
                            st.toast("👍 Thank you for your feedback! The RAG system will boost these sources.")
                        else:
                            st.toast("👎 Thank you for your feedback! The RAG system will penalize or flag these sources.")

# 1. Matched tool-call tracing/logging UI
tool_traces = []
for idx, msg in enumerate(chat_history):
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            # Find matching ToolMessage in subsequent messages
            result_content = "Running..."
            for next_msg in chat_history[idx+1:]:
                if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tc["id"]:
                    result_content = next_msg.content
                    break
            tool_traces.append({
                "name": tc["name"],
                "args": tc["args"],
                "result": result_content
            })
            
if tool_traces:
    with st.expander("🔧 Agent Reasoning & Tool Calls", expanded=False):
        for trace in tool_traces:
            st.markdown(f"🤖 **Tool Executed:** `{trace['name']}`")
            st.markdown("**Arguments:**")
            st.json(trace["args"])
            st.markdown("**Result:**")
            if "INSUFFICIENT" in trace["result"] or "Web Search Results" in trace["result"]:
                st.markdown(trace["result"])
            else:
                st.code(trace["result"])
            st.markdown("---")

# Render last retrieved sources (including transcribed tables or diagrams)
if st.session_state.get("last_retrieved_sources"):
    with st.expander("🔍 Retrieved Sources (Multimodal Context)", expanded=False):
        for src in st.session_state.last_retrieved_sources:
            st.markdown(f"**Source File: `{src['filename']}`** (Similarity: {src['similarity']:.4f})")
            st.markdown(src["content"])
            if src.get("image_data"):
                st.image(src["image_data"], caption=src["filename"], use_container_width=True)
            st.markdown("---")

# 2. Human-in-the-Loop approval/rejection panel
if state.next and "tools" in state.next:
    last_msg = chat_history[-1] if chat_history else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        st.warning("⚠️ **Human-in-the-Loop: Action Requires Approval**")
        st.markdown("The agent is requesting permission to execute the following tool call(s):")
        for tc in last_msg.tool_calls:
            st.markdown(f"👉 **Tool Name:** `{tc['name']}`")
            st.json(tc['args'])
            
        col_app, col_rej = st.columns(2)
        if col_app.button("✅ Approve & Execute", use_container_width=True):
            st.session_state.action = "approve"
            st.rerun()
            
        if col_rej.button("❌ Reject & Deny", use_container_width=True):
            st.session_state.action = "reject"
            st.rerun()

# Prompt input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Clear last retrieved sources before generating new response
    st.session_state.last_retrieved_sources = []
    if "retrieved_sources" in st.session_state:
        st.session_state.retrieved_sources = []
        
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Update thread title in sidebar if it is currently "New Chat"
    for thread in st.session_state.threads:
        if thread["id"] == st.session_state.thread_id and thread["title"] == "New Chat":
            thread["title"] = user_input[:25] + "..." if len(user_input) > 25 else user_input
            break

    # 2. Generator function to stream tokens from LangGraph chatbot
    def stream_response(prompt):
        # We send the new user message as a state update
        state_update = {"messages": [HumanMessage(content=prompt)]}
        for message_chunk, metadata in chatbot.stream(
            state_update,
            config=active_config,
            stream_mode="messages"
        ):
            if message_chunk.content:
                clean_chunk = extract_text_from_content(message_chunk.content)
                if clean_chunk:
                    yield clean_chunk

    # 3. Stream assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(stream_response(user_input))
            # Save current sources as last retrieved sources
            if "retrieved_sources" in st.session_state:
                st.session_state.last_retrieved_sources = st.session_state.retrieved_sources
            # Rerun the app to refresh the state and load conversation correctly
            st.rerun()
        except Exception as e:
            st.error(f"Error calling chatbot: {e}")
            st.info("💡 Make sure to set your GEMINI_API_KEY or GOOGLE_API_KEY in the .env file.")
