import sys
import time
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import UnstructuredMarkdownLoader

# color codes
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

RAG_TEMPLATE = """
You need to answer a question given the following pieces of retrieved context. If you don't know the answer, just say that you don't know. Give as many details as possible, preferably in an outline format with bullet points and headings as needed. If it makes sense, use paragraphs as needed. Make sure that if code is included, it is maintained in your answer and is given preference. Ensure that the content is relevant to the question and is has maximum coverage of the context provided.

<context>
{context}
</context>

Answer the following question:

{question}"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loader = DirectoryLoader("docs/", glob="**/*.md", loader_cls=TextLoader, use_multithreading=True) # show_progress=True
documents = loader.load()
# todo - pre process with llm (put keywords)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
all_splits = text_splitter.split_documents(documents)
# todo - try open ai
local_embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://host.docker.internal:11434")

qdrant = QdrantClient("http://host.docker.internal",port=6333)

try:
    existing_collections = qdrant.get_collections().collections
    collection_names = [col.name for col in existing_collections]
except:
    print(f"Error retrieving collections. Exiting.")
    sys.exit(1)

vectordbname = sys.argv[1] if len(sys.argv) > 1 else "defaultstore"

if vectordbname in collection_names:
    print(f"{BLUE}[*] Collection {vectordbname} already exists. Skipping document addition.{RESET}", flush=True)
    vectorstore = QdrantVectorStore(
        client=qdrant,
        collection_name=vectordbname,
        embedding=local_embeddings,
    )
else:
    qdrant.create_collection(
        collection_name=vectordbname,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=qdrant,
        collection_name=vectordbname,
        embedding=local_embeddings,
    )
    batch_size = 25
    delay = 1.5
    print(f"{BLUE}[+] Generated {len(all_splits)} splits across all documents.{RESET}", flush=True)
    print(f"{BLUE}[+] Proceeding to add {int(len(all_splits) // batch_size)+1} batches of documents.{RESET}", flush=True)
    print(f"{BLUE}[!] This will take at least {int(len(all_splits) // batch_size)*3} seconds.{RESET}", flush=True)
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            print(f"{MAGENTA}{i // batch_size + 1} done! {RESET}", flush=True, end="")
            time.sleep(delay)
        except Exception as e:
            print(f"{RED}Error adding batch {i // batch_size + 1}: {e}{RESET}")
    print(f"\n\n{BLUE}[+] Done adding documents.{RESET}", flush=True)
    print(f"{MAGENTA}--------------------------------------------------------------------------{RESET}\n\n", flush=True)

# keyword_retriever = KeywordRetriever(documents=all_splits)

def hybrid_search(query, vectorstore, k):
    vector_results = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(query)
    # keyword encrichment
    print(f"{RED}Found {len(vector_results)} documents.{RESET}")
    # print(f"{RED}", vector_results, f"{RESET}\n\n")
    return vector_results

def format_hybrid_docs(query):
    docs = hybrid_search(query, vectorstore, k=30)
    return "\n\n".join(doc.page_content for doc in docs)

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.05,
    base_url="http://host.docker.internal:11434",
)

# retriever = vectorstore.as_retriever(search_type="hybrid", search_kwargs={"k": 100})
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
qa_chain = (
    {"context": format_hybrid_docs | RunnablePassthrough(), "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

while True:
    ques = input(f"{BLUE}Ask me a question: {RESET}")
    if ques == "exit":
        break
    output = qa_chain.invoke(ques)
    print("\n\n", output, "\n\n")
    print(f"{MAGENTA}--------------------------------------------------------------------------{RESET}\n\n")
