# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jpassword"

OPENAI_API_KEY = "" # your OpenAI API key
PINECONE_API_KEY = "" # your Pinecone API key
PINECONE_ENV = ""   # example
PINECONE_INDEX_NAME = "travel-hybrid"
PINECONE_VECTOR_DIM = 1536      # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
