import streamlit as st
import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

def init_streamlit_states():
    """Initialize Streamlit session state variables.
    - rag_chain: Stores the RAG chain for question answering
    - models_loaded: Boolean to check if models are loaded.
    - embeddings: Stores the embeddings model.
    - llm: Stores the language model for generating responses.
    """
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "app_name" not in st.session_state:
        st.session_state.app_name = "PDF RAG Assistant App"

init_streamlit_states()

@st.cache_resource
def load_embeddings_model():
    """
    Loads and caches the HuggingFace Vietnamese Biencoder embeddings model for efficient reuse.

    Returns:
    HuggingFaceEmbeddings: An instance of the Vietnamese Bi-encoder embeddings model from HuggingFace.
    """
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_language_model():
    """
    Loads a quantized Vicuna-7B language model and tokenizer with 4-bit NF4 quantization for efficient inference,
    and wraps them in a HuggingFace text-generation pipeline.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model and tokenizer.
    """
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

def processing_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)

    retriever = vector_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(temp_file_path)  # Clean up the temporary file

    return rag_chain, len(docs)

st.set_page_config(page_title=st.session_state.app_name, page_icon=":books:", layout="wide")
st.title(st.session_state.app_name)

st.markdown("""
**Ứng dụng AI giúp bạn hỏi-đáp các nội dụng từ file PDF bằng tiếng Việt.**
**Hướng dẫn sử dụng:**
1. Tải lên file PDF bạn muốn hỏi-đáp.
2. Nhập câu hỏi của bạn vào ô nhập liệu.
""")

# Model loading
if not st.session_state.models_loaded:
    with st.spinner("Đang tải mô hình..."):
        st.session_state.embeddings = load_embeddings_model()
        st.session_state.llm = load_language_model()
        st.session_state.models_loaded = True
    st.success("Mô hình đã được tải thành công!")
    st.rerun()

# File upload
uploaded_file = st.file_uploader("Tải lên file PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Đang xử lý file PDF..."):
        st.session_state.rag_chain, num_docs = processing_pdf(uploaded_file)
    st.success(f"Đã tải {num_docs} đoạn văn bản từ file PDF.")

if st.session_state.rag_chain:
    question = st.text_input("Nhập câu hỏi của bạn:")
    if question:
        with st.spinner("Đang trả lời câu hỏi..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()
        st.write("**Trả lời:**", answer)