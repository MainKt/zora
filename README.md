# zora

## Create a pgvector container
```
docker run -d --name pgvec -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e PGDATA=/var/lib/postgresql/data/pgdata -p 5432:5432 pgvector/pgvector:0.7.2-pg16
```

## Serve Ollama
```
ollama serve
```

## Install dependencies
```
python -m venv env
source ./env/bin/activate
pip install -r requirements.txt
```

## Check it out
```
python rag.py
```
