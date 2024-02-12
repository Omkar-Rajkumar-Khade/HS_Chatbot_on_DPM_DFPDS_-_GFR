import streamlit as st
from haystack import Pipeline
from streamlit_searchbox import st_searchbox

from IPython.display import Image
from pprint import pprint
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
import rich
from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.writers.document_writer import DuplicatePolicy
#from elasticsearch_haystack.document_store import ElasticsearchDocumentStore
#from elasticsearch_haystack.embedding_retriever import ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
# config 
load_dotenv(".env")

# Initialize Elasticsearch document store
document_store = ElasticsearchDocumentStore(hosts=os.environ.get("db_host"),index="apptest1",)

print(document_store.count_documents())

# Initialize preprocessing pipeline
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

# Function to add documents
def add_documents(files):
    # Create the "documents" folder if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")

    for file in files:
        # Save the uploaded file to the "documents" folder
        with open(f"documents/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    # Load documents from the "documents" folder using the preprocessing pipeline
    file_paths = ["documents" / Path(name) for name in os.listdir("documents")]
    preprocessing_pipeline.run({"converter": {"sources": file_paths}})

pprint(len(document_store.filter_documents()))

# pprint(document_store.filter_documents().meta)
# Initialize answer generation pipeline (RAG Pipeline)
generator = HuggingFaceLocalGenerator("mistralai/Mistral-7B-Instruct-v0.1", huggingface_pipeline_kwargs={"device_map":"auto", "model_kwargs":{"load_in_4bit":True, "bnb_4bit_use_double_quant":True, "bnb_4bit_quant_type":"nf4", "bnb_4bit_compute_dtype":torch.bfloat16}}, generation_kwargs={"max_new_tokens": 800})
generator.warm_up()

prompt_template = """<|system|> Using the information contained in the context, give a comprehensive answer to the question.
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

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model_name_or_path="thenlper/gte-large"))
rag_pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)
rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Initialize document retrieval pipeline
doc_retrieval_pipeline = Pipeline()
doc_retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large"))
doc_retrieval_pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=3))
doc_retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# Function to get answer with relevant documents
def get_answer_with_relevant_docs(query):
    # Use the query pipeline to retrieve relevant documents
    relevant_docs = doc_retrieval_pipeline.run({"text_embedder": {"text": query}})['retriever']['documents']

    # Extract relevant document info
    relevant_doc_info = [(doc.id, doc.meta.get("file_path"), doc.meta.get("page_number", "N/A"), doc.content) for doc in relevant_docs]

    # Use the RAG pipeline to generate an answer
    results = rag_pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}})
    answer = results["llm"]["replies"][0]

    return answer, relevant_doc_info

# Streamlit app title
st.title("Question Answering and Document Retrieval")

# Sidebar for uploading files
uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)

# Display uploaded files
if uploaded_files:
    st.sidebar.header("Uploaded Files:")
    for file in uploaded_files:
        st.sidebar.text(file.name)

# Question input
question = st.text_input("Enter your question...")

# Get Answer button
if st.button("Get Answer"):
    if question:
        # Call the function to get answer and relevant documents
        answer, relevant_doc_info = get_answer_with_relevant_docs(question)

        # Display the answer
        st.header("Answer:")
        st.success("Found Answer!")
        print(answer)
        st.write(answer)

        # Display relevant documents
        st.header("Relevant Documents:")
        for doc_id, doc_path, page_number, doc_content in relevant_doc_info:
            st.text(f"Document ID: {doc_id}, Document Path: {doc_path}, Page Number: {page_number}")
            st.write(doc_content)
            st.text("========================== End of Document ============================")

# Upload button to add documents
if st.sidebar.button("Upload Documents"):
    if uploaded_files:
        add_documents(uploaded_files)
        st.sidebar.success("Documents uploaded successfully!")