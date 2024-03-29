import streamlit as st
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import time

if "messages" not in st.session_state:
    welcome_message = """Hi! I'm a chatbot. I can help you get information about Radek and his projects. What would you like to know?"""
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
    
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     full_response = ""
    #     for word in welcome_message.split():
    #         full_response += (word + " ")
    #         message_placeholder.markdown(full_response + "▌")
    #         time.sleep(0.1)
    #     message_placeholder.markdown(full_response)
            
if "client" not in st.session_state:
    st.session_state.client = OpenAI(model="gpt-3.5-turbo-instruct", api_key=st.secrets["OPENAI_API_KEY"], streaming=True)

if "retriever" not in st.session_state:
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'])
    vectordb = FAISS.load_local("faiss_db", embeddings)
    st.session_state.retriever = vectordb.as_retriever()

if "prompt_template" not in st.session_state:
    template = """You are a helpful chatbot that helps people to get information about Radek Szostak. Radek is a data scientist and machine learning engineer.
    You have access to chat history in order to keep context of the conversation. You are also provided with additional documents where you can search for information about Radek and his projects. Chat user don't has access to these documents. Answer with single assistant message at time. Don't predict further user message.

    % start additional documents %
    {doc_context}
    % end additional documents %

    % chat history %
    {chat_context}
    assisstant:
    """
    st.session_state.prompt_template = PromptTemplate(input_variables=['chat_context', 'doc_context'], template=template)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    chat_context = '\n'.join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
    doc_context = "\n\n".join([str(item.metadata) + "\n" + item.page_content for item in st.session_state.retriever.get_relevant_documents(chat_context)])

    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in st.session_state.client.stream(
            st.session_state.prompt_template.format(
                chat_context=chat_context, 
                doc_context=doc_context
            )
        ):
            full_response += (response)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})