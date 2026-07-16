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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

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

# Add an  checkpointer for conversation memory
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





