import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langrapdatabasend import chatbot, generate_thread_id, retrieve_all_threads, delete_thread

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

# Display current message history for the active thread
state = chatbot.get_state(active_config)
chat_history = state.values.get("messages", [])

for msg in chat_history:
    clean_content = extract_text_from_content(msg.content)
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(clean_content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(clean_content)

# Prompt input
user_input = st.chat_input("Ask a question...")

if user_input:
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
            # Rerun the app to refresh the state and load conversation correctly
            st.rerun()
        except Exception as e:
            st.error(f"Error calling chatbot: {e}")
            st.info("💡 Make sure to set your GEMINI_API_KEY or GOOGLE_API_KEY in the .env file.")
