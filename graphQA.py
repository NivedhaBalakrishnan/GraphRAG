import streamlit as st
from graphRAG import GraphRAG
import os




model = GraphRAG()
get_response = model.get_response

# reduce title size
# st.markdown("<h2 style='text-align: center; color: white;'>Acronics Chatbot</h2>", unsafe_allow_html=True)
st.title("Ask me about Inflammation!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

conversation_file_path = "conversation.txt"
# if os.path.exists(conversation_file_path):
#     with open(conversation_file_path, "r") as file:
#         for line in file:
#             role, content = line.strip().split(": ", 1)
#             st.session_state.messages.append({"role": role, "content": content})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get model response
    response = get_response(prompt)

    # Display model response in chat message container
    with st.chat_message("bot"):
        st.markdown(response)
    # Add model response to chat history
    st.session_state.messages.append({"role": "bot", "content": response})

    with open(conversation_file_path, "a") as file:
        file.write(f"user: {prompt}\n")
        file.write(f"bot: {response}\n\n")
