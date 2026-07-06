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
    
    /* Chat message container custom styles */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 80%;
        line-height: 1.5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        display: inline-block;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    
    .chat-bubble-bot {
        background-color: rgba(30, 41, 59, 0.85);
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.05);
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        margin-right: auto;
    }

    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Application header
st.markdown("<h1>Gemini & LangGraph Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A stateful conversational AI built with LangGraph & Google Gemini</p>", unsafe_allow_html=True)

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
    
    # 2. Invoke the compiled LangGraph chatbot
    with st.spinner("AI is thinking..."):
        try:
            # We send the new user message as a state update. 
            # LangGraph checkpointer will load existing history for 'session-1' and append this new message
            state_update = {"messages": [HumanMessage(content=user_input)]}
            result = chatbot.invoke(state_update, config)
            
            # The result['messages'] contains the list of all messages in the state.
            # Get the latest message, which is the AI response.
            ai_response = result["messages"][-1]
            
            # 3. Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(ai_response.content)
                
            # Add assistant message to local session state history
            st.session_state.chat_history.append(ai_response)
            
        except Exception as e:
            st.error(f"Error calling chatbot: {e}")
            st.info("💡 Make sure to set your GEMINI_API_KEY or GOOGLE_API_KEY in the .env file.")
