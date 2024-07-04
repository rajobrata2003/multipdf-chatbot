import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_chroma import Chroma
from htmlTemplates import css, bot_template, user_template, metadata_template, content_template
import os
from langchain_community.document_loaders import PyPDFLoader

counter = 2

def get_pdfs_as_documents(pdf_docs):
    global counter
    documents = []
    for pdf in pdf_docs:
        # Save the uploaded file temporarily
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        # Load and split the PDF
        loader = PyPDFLoader(pdf.name)
        pages = loader.load_and_split()
        documents.extend(pages)
        #print(documents)
        # Remove the temporary file
        os.remove(pdf.name)
        counter = counter+1
    print(counter)
    return documents


def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print("split successful")
    return chunks


def get_vector_store(pdf_docs):

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    vector_store = FAISS.from_documents(documents=pdf_docs, embedding=embeddings)
    # vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    print("vector successful")
    return vector_store


def get_conversation_chain(vector_store):
    global counter
    print("Entering convo")
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.2, "max_length":512, "max_new_tokens":100})
    # llm = HuggingFaceHub(repo_id="lmsys/fastchat-t5-3b-v1.0", model_kwargs={"temperature":0.4, "max_length":450, "max_new_tokens":100})
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        memory=memory)
    
    return conversation_chain


def handle_userinput(user_question):
    global counter
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    
    query = user_question
    docs = st.session_state.vector_store.similarity_search(query, k=10)
    #print(docs)
    
    # Display messages from top to bottom
    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.markdown(":blue-background[**User Input**]")
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.markdown(":red-background[**Response Generated**]")
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            # st.markdown(":green-background[**Raw Data is**]")
            # st.write(system_template.replace(
            #     "{{MSG}}", docs[0].page_content), unsafe_allow_html=True)
            # st.write(system_template.replace(
            #     "{{MSG}}", docs[1].page_content), unsafe_allow_html=True)
            # st.write(docs)
            st.markdown(":green-background[**Raw Data is**]")
            print(len(docs))
            for doc in docs:
                source = doc.metadata.get("source", "N/A")
                page_content_preview = " ".join(doc.page_content.split()[:20])
                st.write(metadata_template.replace("{{MSG}}", f"Source: {source}"), unsafe_allow_html=True)
                st.write(content_template.replace("{{MSG}}",f"Page Content Preview:  {page_content_preview}..."), unsafe_allow_html=True)
                st.write(" ")
    
    
def main():
    global counter
    load_dotenv()
    #os.environ['USE_FAISS_GPU'] = '0'
    st.set_page_config(page_title="MultiPDF", page_icon=":books:",layout="wide")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("MultiPDF")
    user_question = st.text_input("Input field")


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload doc", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                pdf = get_pdfs_as_documents(pdf_docs)
                # st.write(pdf)
                # print(raw_text)
                #get the text chunks    
                # text_chunks = get_text_chunks(raw_text)
                
                # #create the vector store
                st.session_state.vector_store = get_vector_store(pdf)

                #convo chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
    if user_question and st.session_state.vector_store:
        handle_userinput(user_question)
    # if user_question:
    #     handle_userinput(user_question)
                

if __name__ == "__main__":
    
    main()
