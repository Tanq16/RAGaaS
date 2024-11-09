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

# color codes
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

RAG_TEMPLATE = """
You are an assistant tasked with answering a question using the following retrieved context. Follow these guidelines to provide the most accurate and relevant answer:

1. **If you don't know the answer based on the context, explicitly say "I don't know based on the provided context."** Avoid guessing or adding details not found in the context.
2. **Provide information in an organized, hierarchical format**: Use headings, bullet points, and numbering for clear structure, and employ paragraphs where appropriate.
3. **Include all relevant code snippets**: If the context includes code, ensure it is reproduced accurately in the answer.
4. **Focus on relevance**: Only include details directly related to the question. Do not introduce arbitrary or unrelated information from the context.
5. **Avoid redundancy**: Summarize where possible and avoid repeating information unless necessary for clarity.
6. **Acronyms**: For any acronyms you encounter in the query, do not use pre-existing knowledge. Instead, use the context provided to determine the meaning of the acronym.

**Context**:
{context}

**Question**:
{question}

**Answer**:"""

loader = DirectoryLoader("docs/", glob="**/*.md", loader_cls=TextLoader, use_multithreading=True) # show_progress=True
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)
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
    batch_size = 30
    delay = 0.3
    print(f"{BLUE}[+] Generated {len(all_splits)} splits across all documents.{RESET}", flush=True)
    print(f"{BLUE}[+] Proceeding to add {int(len(all_splits) // batch_size)+1} batches of documents.{RESET}", flush=True)
    print(f"{BLUE}[!] This will take at least {int((len(all_splits) // batch_size)*delay)} seconds.{RESET}", flush=True)
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

def hybrid_search(query, vectorstore, k):
    vector_results = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(query)
    # TODO - Add keyword search results
    print(f"{RED}Found {len(vector_results)} documents.{RESET}")
    for doc in vector_results:
        print(f"{BLUE}[*] {doc.metadata}{RESET}")
    return vector_results

def format_hybrid_docs(query):
    docs = hybrid_search(query, vectorstore, k=10)
    returndata = "\n\n".join(doc.page_content for doc in docs)
    return returndata

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
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
