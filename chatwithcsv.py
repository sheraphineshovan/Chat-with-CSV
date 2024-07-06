import os
import gradio as gr
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from io import StringIO

# Load the sentence transformer model
sentence_transformer_model = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer = HuggingFaceEmbeddings(model_name=sentence_transformer_model, model_kwargs={'device': 'cpu'})

# Load the LLaMA-2 model
llm = CTransformers(model="D:\CSVAPP\model\llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0.5)

def llama_response(df, question):
    # Create vector store for the CSV data (for LLaMA-2)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(df.to_string())
    vectorstore = FAISS.from_texts(texts, sentence_transformer)

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    llama_response = chain.invoke({"question": question, "chat_history": []})["answer"]
    return llama_response

def main():
    with gr.Blocks() as demo:
        # Input components
        csv_file = gr.File(label="Upload CSV File")
        question = gr.Textbox(label="Ask a question about the data")

        # Output components
        llama_output = gr.Textbox(label="LLaMA-2 Response")

        # Function to handle the upload and analysis
        def process_data(file, question):
            # Correctly read CSV into a DataFrame
            df = pd.read_csv(StringIO(file))

            # Get LLaMA-2 response
            llama_resp = llama_response(df, question)

            return llama_resp

        # Connect components and function
        gr.Button("Analyze").click(fn=process_data, inputs=[csv_file, question], outputs=[llama_output])

        # Run the interface
        demo.launch(share=True)

if __name__ == "__main__":
    main()