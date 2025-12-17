import os
from dotenv import load_dotenv

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load Environment Variables
load_dotenv()
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')


############################################################################

st.title('PDF RAG Chatbot with Chat History')
st.write("Upload PDFs and Ask Chatbot Questions")

api_key=st.text_input("Enter your Groq API Key", type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model='llama-3.1-8b-instant')


    session_id=st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Upload a PDF File", type="pdf",accept_multiple_files=True)


###########################################################################


    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f"./temp.pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            pdf_loader = PyPDFLoader(temp_pdf)
            docs=pdf_loader.load()
            documents.extend(docs)



        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        huggingface_embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
        vector_store = Chroma.from_documents(chunks,huggingface_embeddings)
        retriever = vector_store.as_retriever()




        contextualised_q_system_prompt=(
            """
                Given a chat history and the latest user question,
                which might reference the context in the chat history,
                formulate a standalone question which can be understood
                without the chat history.

                Do NOT answer the question.
                Just formulate it if needed.
                Otherwise, return it as it is.
            """
        )



        contextualised_q_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualised_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualised_q_prompt)




        # Q&A Prompt
        system_prompt=(
            """
                You are an assistant for consultation tasks.
                Use the following pieces of retrieved context to answer the questions.
                If you don't know, just say that you don't know.
                Keep the answer concise with the maximum of 3 sentences.

                {context}
            """
        )


        qa_prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


        qa_chain = create_stuff_documents_chain(llm,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)




        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )



        user_input = st.text_input("Enter your question:")

        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant: ", response['answer'])
            st.write("Chat History: ", session_history.messages)


    else:
        st.warning("Please enter your Groq API Key")


