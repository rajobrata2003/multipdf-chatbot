# multipdf-chatbot
MultiPDF is a Streamlit-based application that allows users to upload and interact with multiple PDF documents. The application leverages LangChain, OpenAI, and FAISS to create a conversational agent capable of answering questions based on the content of the uploaded PDFs.


## Features

- **PDF Upload**: Upload multiple PDF documents for processing.
- **Text Chunking**: Split the text content of the PDFs into manageable chunks.
- **Vector Store Creation**: Create a vector store using embeddings to facilitate efficient document retrieval.
- **Conversational Interface**: Engage in a conversation with the chatbot to ask questions about the uploaded documents.
- **Document Retrieval**: Retrieve and display relevant document content based on the user's questions.

## Setup

### Prerequisites

- Python 3.7+
- An OpenAI API key
- A .env file containing your OpenAI API key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multipdf-chatbot.git
    cd multipdf-chatbot
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open the application in your web browser. You should see the MultiPDF interface.

3. Use the sidebar to upload your PDF documents.

4. After uploading, click on the "Process" button to process the PDFs.

5. Enter your questions in the input field to interact with the chatbot and retrieve information from the uploaded PDFs.

## Code Overview

### Main Functions

- `get_pdfs_as_documents(pdf_docs)`: Loads and splits the content of the uploaded PDFs into pages.
- `get_text_chunks(text)`: Splits the text into chunks for processing.
- `get_vector_store(pdf_docs)`: Creates a vector store from the PDF documents using embeddings.
- `get_conversation_chain(vector_store)`: Sets up the conversational chain using a language model and a vector store retriever.
- `handle_userinput(user_question)`: Handles user input, retrieves relevant documents, and displays the conversation history.
- `main()`: The main function that initializes the Streamlit app, handles file uploads, and sets up the conversation chain.

### Templates

The application uses HTML templates for styling the chat interface, including user messages, bot responses, and metadata display.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://faiss.ai/)
