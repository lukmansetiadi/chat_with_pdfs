import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = OllamaLLM(model='llama2')
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = OpenAIEmbeddings()
    # embeddings = OpenGroqEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    # user_question = st.text_input("Ask a question about your documents:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        # print("temp")
        handle_userinput(user_question)

    with st.sidebar:
        temp=""
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)


                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # temp = vectorstore

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

        # st.subheader(temp)


if __name__ == '__main__':
    main()