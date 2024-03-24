import streamlit as st
import os
from pathlib import Path
from haystack import Pipeline, Document
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.writers.document_writer import DuplicatePolicy
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
import torch

# Initialize Elasticsearch Document Store
document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200", index="elasticsearchdb")

# Function to initialize and run preprocessing pipeline
def run_preprocessing_pipeline(file_paths, document_store):
    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component("converter", PyPDFToDocument())
    preprocessing_pipeline.add_component("cleaner", DocumentCleaner())
    preprocessing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=400))
    preprocessing_pipeline.add_component("doc_embedder", SentenceTransformersDocumentEmbedder(model="thenlper/gte-large"))
    preprocessing_pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
    preprocessing_pipeline.connect("converter", "cleaner")
    preprocessing_pipeline.connect("cleaner", "splitter")
    preprocessing_pipeline.connect("splitter", "doc_embedder")
    preprocessing_pipeline.connect("doc_embedder", "writer")
    preprocessing_pipeline.run({"converter": {"sources": file_paths}})
    st.sidebar.info("Document processed and stored in Elasticsearch VectorDB!")
    # Exit from the function after preprocessing is complete
    st.stop()

# Function to retrieve answer with relevant documents
def get_answer_with_relevant_docs(query, rag_pipeline, doc_retriever_pipeline):
    relevant_docs = doc_retriever_pipeline.run({"text_embedder": {"text": query}})['retriever']['documents']
    relevant_doc_info = [(doc.id, doc.meta.get("file_path"), doc.meta.get("page_number", "N/A"), doc.content) for doc in relevant_docs]
    
    results = rag_pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query}
    })

    answer = results["llm"]["replies"][0]

    return answer, relevant_doc_info

# Function to initialize and run RAG pipeline
def initialize_rag_pipeline():
    prompt_template = """ <|system|> Using the information contained in the context, give a comprehensive answer to the question.
            If the answer is contained in the context, also report the source data.
            If the answer cannot be deduced from the context, do not give an answer.</s>
            <|user|>
            Context:
            {% for doc in documents %}
            {{ doc.content }} 
            {% endfor %};
            \nQuestion: {{query}}
            </s>
            
            Example:
            \nQuestion:What are the documents and stages for undertaking an open tender enquiry under DFPDS schedule 19.3?
            \nAnswer: 1. In Principal Approval (IPA):
               - What happens: Getting the initial approval to proceed with the procurement.
               - Documents involved: Explanation of the project, what needs to be done, how much it will cost, and any financial support needed.
               - Key point: This approval comes from the Central Financial Assistance (CFA) after proper discussions.

            2. Acceptance of Necessity (AoN):
               - What happens: Confirming that there's a genuine need for the procurement and estimating the costs.
               - Documents involved: Cost estimates, comparisons with previous purchases, and making sure the plan is clear.
               - Key point: The CFA approves this, considering suggestions from the Independent Financial Advisor (IFA) if needed.

            3. Upload Bid on GeM:
               - What happens: Putting the details of the procurement online, especially on the Government e-Marketplace (GeM).
               - Key point: Timing depends on the type of procurement.

            4. Technical Evaluation Committee (TEC):
               - What happens: Checking if the bidders meet the technical requirements.
               - Documents involved: Scrutinizing documents submitted by bidders and ensuring they meet the specified criteria.
               - Key point: Non-compliant bidders get a chance to fix issues within 48 hours.

            5. Benchmarking:
               - What happens: Comparing the estimated costs with market prices to ensure fairness.
               - Documents involved: Checking how prices compare to what was estimated.
               - Key point: Prices are set a bit higher than the initial estimates.

            6. Commercial Opening of Bid (COB):
               - What happens: Opening bids to see the prices submitted by compliant bidders.
               - Key point: Analysis of prices helps decide if negotiation is needed with the top bidder. Reverse auction may happen if multiple bidders qualify.

            7. Cost Negotiation Committee (CNC)/Price Negotiation Committee (PNC):
               - What happens: Checking if the costs can be negotiated and if it aligns with guidelines.
               - Key point: Negotiations are done, and financial details are recommended for approval. All this is reviewed by the CFA.

            8. Expenditure Angle Sanction (EAS):
               - What happens: Getting the final approval for the expenses from the CFA.
               - Key point: This is the green light to go ahead with the procurement.

            9. Supply Order:
               - What happens: Placing the order with the chosen bidder.
               - Key point: The procurement process is complete, and the chosen bidder is officially hired to do the job.

            <|assistant|>
            \nAnswer:
            """
    prompt_builder = PromptBuilder(template=prompt_template)
    # Initialize answer generation pipeline (RAG Pipeline)
    generator = HuggingFaceLocalGenerator("mistralai/Mistral-7B-Instruct-v0.1",
                                        huggingface_pipeline_kwargs={"device_map": "auto",
                                                                    "model_kwargs": {"load_in_4bit": True,
                                                                                    "bnb_4bit_use_double_quant": True,
                                                                                    "bnb_4bit_quant_type": "nf4",
                                                                                    "bnb_4bit_compute_dtype": torch.bfloat16}},
                                        generation_kwargs={"max_new_tokens": 700})
    generator.warm_up()
    
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large"))
    rag_pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=3))
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    return rag_pipeline

# Function to initialize and run document retriever pipeline
def initialize_doc_retriever_pipeline():
    doc_retriever_pipeline = Pipeline()
    doc_retriever_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large"))
    doc_retriever_pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=3))
    doc_retriever_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return doc_retriever_pipeline

# Main Streamlit UI
st.title("Chatbot On DPM, DFPDS & GFR documents")

# Check if the LLM (Language Model) is already loaded
if 'rag_pipeline' not in st.session_state:
    # Load LLM
    st.session_state.rag_pipeline = initialize_rag_pipeline()

# Sidebar
st.sidebar.title("Document Upload")
upload_doc = st.sidebar.radio("Do you want to upload a document?", ["Yes", "No"])

# Check if the user wants to upload a document
if upload_doc == "Yes":
    uploaded_files = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        # Create a folder for uploaded documents if it doesn't exist
        Path("Uploaded_documents3").mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded files to the "Uploaded_documents2" folder
        file_paths = []
        for file in uploaded_files:
            file_path = Path("Uploaded_documents3") / Path(file.name)
            file_path.write_bytes(file.getvalue())
            file_paths.append(file_path)

        run_preprocessing_pipeline(file_paths, document_store)

# Ask the user for a query
query = st.text_input("Ask a question:")

# Submit button to trigger QA
if st.button("Get Answer"):
    if query:
        # Retrieve answer with relevant documents using the RAG pipeline and document retriever pipeline
        answer, relevant_doc_info = get_answer_with_relevant_docs(query, st.session_state.rag_pipeline, initialize_doc_retriever_pipeline())

        # Display the answer
        Answer_text = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Answer:</p>'
        st.markdown(Answer_text, unsafe_allow_html=True)
        #st.write("Answer:")
        st.write(answer)

        # Display relevant documents with page numbers
        Relevant_docs_text = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Relevant Documents:</p>'
        st.markdown(Relevant_docs_text, unsafe_allow_html=True)
        #st.write("Relevant Documents:")
        for doc_id, doc_path, page_number, doc_content in relevant_doc_info:
            st.write(f"Document ID: {doc_id}, Document Path: {doc_path}, Page Number: {page_number}")
            st.write("-------------------------------------------------------------------------------")
            doc_content_text = '<p style="font-family:sans-serif; color:Purple; font-size: 20px;">Document Content:</p>'
            st.markdown(doc_content_text, unsafe_allow_html=True)
            st.write(f"\n{doc_content}")
            st.write("========================== End of Document ================================================")
   
# Stop button to terminate the app
stop_app = st.sidebar.button("Stop App")
if stop_app:
    st.sidebar.warning("Stopping the application. Please close the browser tab.")
    os._exit(0)