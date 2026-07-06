import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import chatbot, config

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

# Initialize message history in streamlit session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display current message history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Prompt input
user_input = st.chat_input("Ask a question...")

if user_input:
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add human message to local session state history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # 2. Generator function to stream tokens from LangGraph chatbot
    def stream_response(prompt):
        # We send the new user message as a state update
        state_update = {"messages": [HumanMessage(content=prompt)]}
        for message_chunk, metadata in chatbot.stream(
            state_update,
            config=config,
            stream_mode="messages"
        ):
            if message_chunk.content:
                yield message_chunk.content

    # 3. Stream assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(stream_response(user_input))
            # Add assistant message to local session state history
            st.session_state.chat_history.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"Error calling chatbot: {e}")
            st.info("💡 Make sure to set your GEMINI_API_KEY or GOOGLE_API_KEY in the .env file.")
