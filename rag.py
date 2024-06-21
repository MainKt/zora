import asyncio
import itertools
from RealtimeTTS import TextToAudioStream, CoquiEngine

from pgvector.psycopg import register_vector_async
import psycopg
from sentence_transformers import CrossEncoder
from pypdf import PdfReader
import os
import argparse
import ollama

DB_URL = os.getenv('DB_URL', 'postgresql://postgres:postgres@localhost:5432')
async def get_db():
    conn = await psycopg.AsyncConnection.connect(DB_URL, autocommit=True)
    return conn

MODEL = 'mistral'
EMBEDDING_MODEL = 'mxbai-embed-large'
EMBEDDING_SIZE = 1024
DATA_DIR = 'data'

async def create_schema(conn):
    await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    await register_vector_async(conn)

    await conn.execute('DROP TABLE IF EXISTS documents')
    await conn.execute(f'CREATE TABLE documents (id bigserial PRIMARY KEY, source text, content text, embedding vector({EMBEDDING_SIZE}))')
    await conn.execute("CREATE INDEX ON documents USING GIN (to_tsvector('english', content))")


async def insert_data(conn, documents):
    contents = [document['content'] for document in documents]
    sources = [document['source'] for document in documents]

    embeddings = [ollama.embeddings(model=EMBEDDING_MODEL, prompt=content)["embedding"] for content in contents]

    sql = 'INSERT INTO documents (source, content, embedding) VALUES ' + ', '.join([f'(%s, %s, %s::vector({EMBEDDING_SIZE}))' for _ in embeddings])
    params = list(itertools.chain(*zip(sources, contents, embeddings)))
    await conn.execute(sql, params)


async def semantic_search(conn, query):
    embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)["embedding"]

    async with conn.cursor() as cur:
        await cur.execute(f'SELECT source, content FROM documents ORDER BY embedding <=> %s::vector({EMBEDDING_SIZE}) LIMIT 5', (embedding,))
        return await cur.fetchall()


async def keyword_search(conn, query):
    async with conn.cursor() as cur:
        await cur.execute("SELECT source, content FROM documents, plainto_tsquery('english', %s) query WHERE to_tsvector('english', content) @@ query ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC LIMIT 5", (query,))
        return await cur.fetchall()


def rerank(query, results):
    # deduplicate
    results = set(itertools.chain(*results))

    # re-rank
    encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = encoder.predict([(query, item[1]) for item in results])
    return [v for _, v in sorted(zip(scores, results), reverse=True)]

def read_pdfs():
    documents = []
    for filename in os.listdir(DATA_DIR):
        reader = PdfReader(DATA_DIR + '/' + filename)
        number_of_pages = len(reader.pages)
        content = ''
        for page_number in range(number_of_pages):
            page = reader.pages[page_number]
            content += page.extract_text()
        documents.append({'source': filename, 'content': content})
    return documents

def read_txts():
    return [
        { 'source': filename, 'content': open(os.path.join(DATA_DIR, filename), 'r').read() }
        for filename in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, filename))
    ]

def system_prompt(context):
    system = f"""
    Instructions:
    - Be short and concise and less than two sentences.
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
    - Cite your sources
    Context: {context}
    """

    alternate = f"""
    Instructions:
    - Be short and concise and less than two sentences.
    """

    return system if len(context) != 0 else alternate

def prompt_ollama(query, contexts):
    stream = ollama.generate(
        model=MODEL,
        prompt=query,
        stream=True,
        system=system_prompt(contexts)
    )

    return stream

def print_ollama_stream(stream):
    for chunk in stream:
      print(chunk['response'], end='', flush=True)

def tts(ollama_stream):
    engine = CoquiEngine()
    stream = TextToAudioStream(engine)
    for chunk in ollama_stream:
        stream.feed(chunk['response'])
    stream.play_async()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='The query prompt')
    parser.add_argument('--prompt', action='store_true', help='Prompt Ollama')

    args = parser.parse_args()
    query = args.query

    documents = read_pdfs()
    # documents.extend(read_txts())

    conn = await get_db()
    await create_schema(conn)
    await insert_data(conn, documents)

    # perform queries in parallel
    contexts = await asyncio.gather(semantic_search(conn, query), keyword_search(conn, query))
    contexts = rerank(query, contexts)

    print('-' * 10 + 'CONTEXTS' + '-' * 10)
    print(contexts)
    print('-' * 10 + 'END OF CONTEXTS' + '-' * 10)

    if args.prompt:
        print('LLM RESPONSE:')
        response_stream = prompt_ollama(query, contexts)
        print_ollama_stream(response_stream)
        tts(response_stream)


if __name__ == "__main__":
    asyncio.run(main())
