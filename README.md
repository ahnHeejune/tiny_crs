# tiny_crs
Tiny CRS (conversational recommender system) 
(c) 2025 Shopvisor Inc. 

## Technology used 
- Langchain 
- HugginFace transformers (local embedding model)
- OpenAI (you need setup OPENAI_API_KEY)
- FAISS
- SQLite 
- Flask

## Database used 
- From Kaggle Shoes dataset 
- https://www.kaggle.com/datasets/mdwaquarazam/shoe-dataset

## How to run
- console:  python tiny_crs.py   (two options: recommendation and full crs service)
- web:      python crs_flask.py  (port number is 5002)  and access using a browser

## Internal Structure
- 2 agents: salesperson and manager
- hybrid DB and RAG: SQL and semantic 

```
                Ajax                +-------------+
 customer <----------------*------->| sales agent | ----------------------------- LLM
               (flask)     |        +-------------+ 
                           |               | conversation history
                           |        +------v-------+ -----------------------------LLM
                           +--------| manager agent| ------------> SQL DB  
                                    +--------------+              Vector DB

```
