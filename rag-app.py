import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub



Loader = PyPDFLoader
FILE_PATH = "./YOLOv10_Tutorials.pdf" # Replace with your PDF file path
loader = Loader(FILE_PATH)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    buffer_size=1,                          # Nhóm 3 câu
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    # number_of_chunks= 30,                   # Số chunks mong muốn. Tìm number_of_chunks - 1 điểm có similarity thấp nhất để cắt. Nếu không đủ điểm cắt phù hợp → số chunks thực tế ≠ số chunks mong muốn
    min_chunk_size=500,                     # Tối thiểu 1000 ký tự
    # sentence_split_regex=r'(?<=[.?!…])\s+', # Bao gồm dấu … cho tiếng Việt
    add_start_index=True
)

docs = semantic_splitter.split_documents(documents)
print("Number of semantic chunks: ", len(docs))

vector_db = Chroma.from_documents(documents=docs,
                                  embedding=embeddings)

retriever = vector_db.as_retriever()

QUERY = "YOLOv10 dùng để làm gì"
result = retriever.invoke(QUERY)

print("Number of relevant documents: ", len(result))

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="cpu"
)

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
output

answer = output.split('Answer:')[1].strip()
answer