import os
import uuid
import time
from flask import Flask, request, render_template, jsonify, make_response
import tiny_crs 

# Session Managment 
SESSION_TIMEOUT = 60 * 30  # 30분
session_data = {} # user_id to state

app = Flask(__name__)

def init_system():

    ''' setup LLM, SQL chain, VectorDB '''
    global salesperson, manager 
   
    # dataset
    csv_path = "MEN_SHOES.csv"
    table_name = "shoes"
    sqldb_path = "shoes.db"
    vectdb_path = "shoes_faiss"

    tiny_crs.init_crs(sqldb_path, vectdb_path)

@app.route("/")
def index():
    global session_data

    # session 
    user_id = request.cookies.get('user_id')
    if not user_id or user_id not in session_data:
        user_id = str(uuid.uuid4())
        salesperson = tiny_crs.create_salesperson()
        manager  = tiny_crs.create_manager()
        session_data[user_id] = {'salesperson': salesperson, 'manager': manager, 'history': [], 'last_active': time.time()}
        print(f"create a new session for {user_id}")

    resp = make_response(render_template('index.html'))
    resp.set_cookie('user_id', user_id, httponly=True)
    return resp

    #return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global session_data

    # 1. input processing 
    user_id = request.cookies.get('user_id')
    print(f"user_id:{user_id}")
    if not user_id or user_id not in session_data:
        return redirect('/')  # 또는 make_response(redirect('/')) 도 가능

    s = session_data[user_id]

    # 2. input processing 
    user_input = request.json["message"]

    # 3. handle dialog 
    salesperson, manager = s['salesperson'], s['manager']
    answer, search_results = tiny_crs.serve_customer(user_input, salesperson, manager)

    # 4. response 
    if search_results is  None:
        search_results = []

    return jsonify({
       "response": answer,
       "results": search_results
    })

@app.route("/detail/<item_id>")
def detail(item_id):
    item = df[df["id"].astype(str) == str(item_id)].to_dict(orient="records")
    if item:
        return jsonify(item[0])
    return jsonify({"error": "Item not found"}), 404

@app.route('/reset', methods=['POST'])
def reset():
    user_id = request.cookies.get('user_id')
    if user_id in global_session_data:
        global_session_data[user_id]['history'] = []
    return jsonify({'status': 'reset'})


# 오래된 세션 삭제 (예: 주기적으로 호출)
def cleanup_sessions():
    now = time.time()
    to_delete = [uid for uid, data in global_session_data.items()
                 if now - data['last_active'] > SESSION_TIMEOUT]
    for uid in to_delete:
        del global_session_data[uid]

# RUN APP
if __name__ == '__main__':  

    #1. init system 
    init_system()
    
    #2. start web service
    print("launching crs web service....")
    app.run('0.0.0.0', port=5002, debug=False)
