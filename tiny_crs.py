'''
   Tiny CRS (Conversational Recommendation System) 
   using Language Models and Agent framework

   (c) 2025 Shopvisor Inc.  heejune@seoultech.ac.kr 

   - Hugginface Local Embdedding model 
   - OpenAI LLM
   - Langchain  
'''

import os
from typing import Union
import logging
logging.basicConfig( level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s'  # NO datefmt!
    )
logger = logging.getLogger("tiny_crs")

####################################################
# DB Module  
####################################################
import pandas as pd

def print_table(sqlfile_path:str, table_name:str, n:int) -> None:

    ''' print first n rows for checking '''

    import sqlite3
    conn = sqlite3.connect(sqlfile_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    for i, row in enumerate(rows):
        if i >= n:
            break
        logger.info(str(row))

    # Close the connection
    conn.close()

def cvt_csv2sqldb(csv_path:str, sql_path:str, table_name:str) -> None:

    ''' 
       convert to csv file into sqllib db 
    ''' 

    # 1. load csv to df 
    import pandas as pd
    df = pd.read_csv(csv_path)
    logger.info(f"Column names:{df.columns.tolist()}")
    logger.info(f"Number of rows:{len(df)}")
    logger.info("samples:")
    logger.info(f"{df.head(3)}")

    # 2. convert df
    if True:
        # 1. change the column names 
        df.columns = ['brand', 'sales', 'price', 'model', 'rating'] 
        # 2.1 remove the Luppy currency symbol and comma 
        df['price'] = df['price'].str.replace('₹', '', regex=False).str.replace(',', '')  
        # 2.2 change the currency and data type  
        df['price'] = df['price'].astype(float) / 10
        # 2.3 add currency column
        df['currency'] = 'USD'
        logger.info("samples converted:")
        logger.info(f"{df.head(3)}")

    # 3. insert df into SQLite as a table 
    import sqlite3
    conn = sqlite3.connect(sql_path)
    df.to_sql(table_name, conn, if_exists="replace", index=True)
    conn.close()
    # verify 
    print_table(sql_path, table_name, 3)

def cvt_df2vectdb(df, embedding, vectdb_dir_path:str) -> None: 

    ''' 
     convert df file to faiss vector db 
     df : comes from sql 
    '''

    # 1. Document list (page_content + meta)
    from langchain.schema import Document
    docs = []
    for _, row in df.iterrows():
        # TEMP, include all for embedding 
        model = row.get("model", "")    
        brand = row.get("brand", "")
        price = row.get("price", "")
        sales = row.get("sales", "")
        rating = row.get("rating", "")
        currency  = row.get("currency", "")    

        if "details" in row:
            text = row["details"]
        else:
            text = (
                f"model: {model}. "
                f"brand: {brand}. "
                f"price: {currency} {price}. "
                f"rating: {rating}. "
                f"sales: {sales}."
            )
        
        metadata = {
            "id": row["index"],
            "model": model,
            "brand": brand,
            "price": price,
            "currency": currency,
            "sales": sales,
            "rating": rating,
            "source": f"product_{row.get('id') or row.get('index')}"  
        }
        docs.append(Document(page_content=text, metadata=metadata))

    # 2. create FAISS DB
    from langchain.vectorstores.faiss import FAISS
    embedding = create_embedding()
    faiss_index = FAISS.from_documents(docs, embedding)

    # 3. save db index 
    faiss_index.save_local(vectdb_dir_path)
   
def prepare_dataset(csv_path:str, sqldb_path:str, table_name:str, vectdb_path:str) -> None:

    ''' csv to sql db and vector db '''

    # 1. from csv to database 
    cvt_csv2sqldb(csv_path, sqldb_path, table_name)

    # 2. sql table to vector db 
    import pandas as pd
    import sqlite3
    conn = sqlite3.connect(sqldb_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logger.info(f"{df.columns.tolist()}")

    embedding = create_embedding()
    cvt_df2vectdb(df, embedding, vectdb_path)

#################################################################
# 
#################################################################
def create_embedding(embedding_model_name:str = "BAAI/bge-small-en-v1.5"):
 
    if embedding_model_name == "openai": 
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        import torch
        encode_kwargs = {'normalize_embeddings': True}  # True for cosine similarity
        embedding = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs=encode_kwargs
        )
    return embedding

_workflow_map, _llm, _sql_chain, _retriever, _embedding = None, None, None, None, None

def init_crs(sqldb_path:str, vectdb_path:str, llm_model_name:str = "gpt-4-turbo"):

    global _workflow_map, _llm, _sql_chain, _retriever, _embedding

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if len(api_key) == 0:
         info.critial("Please set OPENAI_API_KEY")
         exit()

    # 1. CRS workflow 
    if _workflow_map is None:
        _workflow_map = {
            "스타일 기반 탐색": run_style_flow,
            "문제 해결형 탐색": run_problem_solving_flow,
            "후기 기반 탐색": run_review_flow_graph
        }
    
    # 2. LLM
    if _llm is None:
        from langchain_openai import ChatOpenAI #from langchain.llms import ChatOpenAI
        _llm =  ChatOpenAI(temperature = 0, model_name = llm_model_name )

    # 3. SQL Chain     
    if _sql_chain is None:
        from langchain_experimental.sql import SQLDatabaseChain #from langchain.chains import SQLDatabaseChain
        #from langchain.chains.combine_documents.stuff import StuffDocumentsChain
        from langchain.sql_database import SQLDatabase
        db = SQLDatabase.from_uri(f"sqlite:///{sqldb_path}")
        _sql_chain = SQLDatabaseChain.from_llm(_llm, db, verbose = True)
  
    # 4. Vector DB
    if _retriever is None:
        from langchain.vectorstores.faiss import FAISS
        _embedding = create_embedding()
        faiss_index = FAISS.load_local(vectdb_path, _embedding, allow_dangerous_deserialization=True)
        _retriever = faiss_index.as_retriever(search_kwargs={"k": 20})

    #return _workflow_map, _llm, _sql_chain, _retriever, _embedding

# Note: at preset, we donot use two phase approach.
# simply using single, but muti-agent based process 
# sale agent consults a client for preferences and manager monitor their discussion to decide time to recommend  
############################################################################
# Phase 1: conversation type  decision  
############################################################################
def choose_workflow(user_input:str, workflow_map:dict):

    ''' check the intent and run the corresponding workflow '''    
    
    from langchain_core.runnables import Runnable, RunnableMap
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 1. intent detection
    intent_prompt = PromptTemplate.from_template("""
"{question}" 이 질문은 어떤 추천 흐름에 적합한가요?
다음 중 하나로 대답하세요: 스타일 기반 탐색, 문제 해결형 탐색, 후기 기반 탐색
""")
    intent_chain = intent_prompt | llm | StrOutputParser()

    intent = intent_chain.invoke({"question": user_input})
    selected_graph = workflow_map.get(intent, fallback_graph)
        
    return selected_graph

def generate_followup_questions(intent_description: str):

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    chain = LLMChain(llm=llm, prompt=followup_prompt_template)
    q_text = chain.run(intent_description)
    questions = [q.strip() for q in q_text.strip().split("\n") if q.strip()]
    return questions

def extract_user_preferences_from_conversation(user_dialog: str):

    result = extract_chain.run(user_dialog)
    preferences = parser.parse(result)
    return preferences

############################################################################
# Phase 2: main requirement collection flow
############################################################################
def run_problem_solving_flow(user_input):

    # Step 1: You already detected the intent, so we proceed
    logger.info("AI: OK, I got you. Please let me ask a few more questions.")

    # Step 2: Generate follow-up questions
    questions = generate_followup_questions("사용자가 특정 발 문제 (예: 발볼 넓음, 통풍 필요, 쿠션 필요)를 해결하기 위한 신발을 찾고 있습니다.")
    answers = []
    for q in questions:
        logger.info(f"AI: {q}")
        user_input = input("You: ")
        answers.append(f"{q}\n{user_input}")

    full_dialog = "\n".join(answers)

    # Step 3: Extract preferences
    preferences = extract_user_preferences_from_conversation(full_dialog)
    return preferences

############################################################################
# Phase 3: Recommedation 
############################################################################

def intersect(filtered_df, semantic_docs):
    ''' 
       filtering both matching vector db search and SQL search   
       filtered_df : SQL filtered list 
       sematic_docs: Document form vector DB 
    '''
    
    # 1. get index list of SQL results
    logger.debug(f"filtered column:{filtered_df.columns}")  
    df_ids = filtered_df["index"].tolist()  
    logger.debug(f"df_ids:{df_ids}")

    # 2. filtering DOCs using the list 
    filtered_docs = [doc for doc in semantic_docs if doc.metadata['id'] in df_ids ]
    logger.debug(f"filtered docs:{filtered_docs}")

    return filtered_docs
   
def recommend_products(search_criteria, debug = False):

    ''' 
        SQL filtering + Vector DB filtering => RAG (LLM) => result 

        AI reply with recommendations
    '''

    global _llm, _sql_chain, _retriever, _embedding

    logger.debug(f"SEARCH CRITERIA:{search_criteria}")
    #logger.debug("TYPE(SQL_CHAIN):{type(_sql_chain)}")
   
    # 1. SQL search 
    sql_qna = _sql_chain.invoke(search_criteria + ". Please include the product id. Return only the raw SQL, without markdown formatting or explanation.")  # ID explicitely
    if debug:
        logger.info(f"sql_qna:{type(sql_qna)}, {sql_qna}")
    # Extract SQL query string
    query_str = sql_qna["result"].split("SQLQuery:")[-1].strip()
    query_str.strip().removeprefix("```sql").removesuffix("```").strip("`").strip()
   
    from langchain_community.utilities import SQLDatabase
    import pandas as pd
    #2. Run the query using LangChain's SQLDatabase
    db = _sql_chain.database
    #df = db.run_no_throw(query_str)  # returns string
    #logger.info(f"Raw string result:{type(df)}\n{df}")
    with db._engine.connect() as conn:
         sql_results_df = pd.read_sql(query_str, conn)
         logger.debug(f"sql_results_df:{sql_results_df}")
    
    # 2. semantic search 
    semantic_docs = _retriever.get_relevant_documents(search_criteria)
    if debug:
        logger.info(f"semantic_docs:{semantic_docs}")
    
    # 3. Combine two results
    intersect_docs = intersect(sql_results_df, semantic_docs)

    # 4. RAG from two DB  
    from langchain.schema import Document    
    from langchain.vectorstores.faiss import FAISS
    #docs = [Document(page_content=item["description"], metadata=item) for item in intersect_docs]
    faiss_index = FAISS.from_documents(intersect_docs, _embedding)
    retriever2 = faiss_index.as_retriever()
    from langchain.chains import RetrievalQA
    rag_chain = RetrievalQA.from_chain_type(_llm, retriever=retriever2)
    response = rag_chain.run("다음조건을 가장 잘 만족하는 상품을 순서대로 3개까지만 골라줘."
                             "간락한 추천이유도 포함해서."
                             f"선택조건:{search_criteria}")

    shoes_list = [doc.metadata for doc in intersect_docs]  
    return response, shoes_list 

def run_style_flow():
    pass

def run_problem_solving_flow():
    pass

def run_review_flow_graph():
    pass

def serve_customer(user_input:str, salesperson, manager):
     
    global _workflow_map, _llm, _sql_chain, _retriever, _embedding

    sales_reply = salesperson.invoke(user_input)

    manager_reply = manager.run(salesperson.memory.buffer)
    logger.info(f"manager:{manager_reply}")
    if manager_reply.startswith("Yes"):
        # 1.extract the criteria 
        import re
        match = re.search(r"json\s*(\{.*?\})\s*", manager_reply, re.DOTALL)
        if match:
            json_str = match.group(1)
            #data = json.loads(json_str)
            #result = json.dumps(data, ensure_ascii=False, indent=2)
            search_crit = json_str
        else:
            search_crit = manager_reply[3:] # when no json marking  

        # 2. search recomendations
        final_reply, shoes_list = recommend_products(search_crit, debug = True)
        final_reply += "\n맘에 드시는 제품이 있으신가요? 아니면 다른 신발을 보여 드릴까요?" 
        salesperson = create_salesperson()
        return final_reply, shoes_list
    else:
        return sales_reply['text'], [] 

def create_salesperson():

    ''' Salesperson Agent, Prompt with chat history and new cutomer input '''

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    #from langchain.schema import HumanMessage
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
    sales_prompt = PromptTemplate.from_template( 
    """ 당신은 신발을 추천해주는 전문가입니다. 신발을 신을손님의 다음과같은 정보를 확보해야 추천을 할 수있습니다.
용도, 성별, 선호브랜드, 사이즈, 발볼 (좁은지, 넖은치, 보통인지), 예산 
말은 길지않게 간략하게 하도록 하고, 물어 볼것은 한꺼번에 물어 보지 말고 하나씩 차례대로 물어 보세요. 
제품에 대한 일반적인 설명은 해도 좋지만 최종 제품 추천은 하지 마세요.
아래는 고객과의 이전 대화입니다:
{chat_history}
현재 질문: {input}
필요한 정보를 얻기 위해 적절한 답변 또는 질문을 이어가세요.""")

    sales_chain = LLMChain(llm=_llm, prompt=sales_prompt, memory=memory)

    return sales_chain

def create_manager():

    ''' Manager Agent: Prompt with chat history between the customer and salesperson '''

    from langchain_core.prompts.chat import ChatPromptTemplate
    manager_prompt = ChatPromptTemplate.from_messages([
                    ("system", '''당신은 신발매장의 관리자이다. 
당신의 역할은 직원이 고객과 상담하는 것을 듣고 신발 추천에 필요한 충분한 선호도가 모아 졌는지 판단하는 일이다.
기준은 용도, 성별, 선호브랜드, 사이즈, 발볼 (좁은지, 놃은치, 보통인지), 예산이다.
만약 고객이 특정하지 않는 기준이 있다면 이는 선호도가 없는 것으로 한다.'''), 
                    ("human", "다음은 지금까지 직원과 고개의 대회내용이다.\n"
                              "{chat_history}\n"
                              "이를 바탕으로 다음과 같이 답하라.\n"
                              "신발을 추천할 충분한 조건이 확보되었으면 'Yes' 라고 응답하고, 요구사항을 Json 형대로 반환하라. 확보도지 않은 것은 미확보라고 표시하라.\n"
                              "신발을 추천할 조건이 확보되지 않았으면 'No' 라고 응답하고, 요구사항을 Json 형대로 반환하라. 확보도지 않은 것은 미확보라고 표시하라.\n"
                              )])

    from langchain.chains import LLMChain
    manager_chain = LLMChain(llm=_llm, prompt=manager_prompt)
   
    return manager_chain
     
if __name__ == "__main__":


    # dataset 
    csv_path = "MEN_SHOES.csv"
    table_name = "shoes" 
    sqldb_path = "shoes.db" 
    vectdb_path = "shoes_faiss"

    if not os.path.isdir(vectdb_path): # remove the directory if you want recreate 
        prepare_dataset(csv_path, sqldb_path, table_name, vectdb_path)
    
    init_crs(sqldb_path, vectdb_path)

    test_case = 1
    if test_case == 1:  # Hybrid Recomendation Test

        search_criteria = "Adidas, sold more than 1000, less than 200 dollars"
        reply, shoes_list = recommend_products(search_criteria, debug = False)
        logger.info(f"Shopvisor:{reply}")
        logger.info("#shoes list#")
        for i, shoes in enumerate(shoes_list):
            if i == 0:
                col_str = ", ".join(str(k) for k in shoes.keys())
                logger.info(col_str) 
            val_str = ", ".join(str(v) for v in shoes.values())
            logger.info(val_str) 

    elif test_case == 2:  #  CRS Full Test 

        # create agents
        salesperson = create_salesperson()
        manager = create_manager()

        # AI starts the conversation, not a human 
        user_input = input("점원: 안녕하세요 손님, 어떤 신발을 보고 계세요?\n손님:") 

        while True:

             # the salesperson leads the dialog 
             sales_reply = salesperson.invoke(user_input)

             # the manager monitors the dialog  
             manager_reply = manager.run(salesperson.memory.buffer)
             logger.info(f"Manager:{manager_reply}")
             if manager_reply.startswith("Yes"):

                  # extract the criteria 
                  import re
                  match = re.search(r"json\s*(\{.*?\})\s*", manager_reply, re.DOTALL)
                  if match:
                      json_str = match.group(1)
                      #data = json.loads(json_str)
                      #result = json.dumps(data, ensure_ascii=False, indent=2)
                      search_crit = json_str
                  else:
                      search_crit = manager_reply[3:] # when no json marking  

                  # recomendations
                  reply, shoes_list = recommend_products(search_crit, debug = True)
                  
                  # print recommendation list
                  for i, shoes in enumerate(shoes_list):
                      if i == 0:
                          col_str = ", ".join(str(k) for k in shoes.keys())
                          logger.info(col_str) 
                      val_str = ", ".join(str(v) for v in shoes.values())
                      logger.info(val_str) 
                
                  salesperson = create_salesperson()
                  user_input = input("점원: 안녕하세요 손님, 어떤 신발을 보고 계세요?\n손님:") 

             else: # continue sales dialog 
                  user_input = input(f"점원:{sales_reply['text']}\n손님:") 
              
