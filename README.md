# RAGaaS

With Ollama running and the required models (`mxbai-embed-large` and `llama3.1`) are pulled, the image can be built with &rarr;

```bash
docker builld -t testrag .
```

Then start a qdrant container with &rarr;

```bash
docker run -p 6333:6333 -v qdrantdata:/qdrant/storage qdrant/qdrant
```

Lastly, run the image with &rarr;

```bash
docker run --rm -v /path/to/your/knowledgebase:/app/docs -i testrag knowledgebasedb
```

Code explanation in [companion blog post](https://tanishq.page/blog/posts/ai-rag/).
