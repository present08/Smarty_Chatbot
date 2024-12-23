from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import socket
import json

from config.DatabaseConfig import *
from utils.Database import Database

host = "127.0.0.1"
port = 5050

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

def get_answer_from_engine(bottype, query):
    mySocket = socket.socket()
    mySocket.connect((host, port))

    json_data = {
        'Query' : query,
        'BotType' : bottype
    }
    message = json.dumps(json_data)
    mySocket.send(message.encode())

    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)

    mySocket.close()

    return ret_data

db = Database(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME, charset='utf8mb4')
conn = db.connect()
def save_data(messages, time, user_id):
    print("DB에 들어갈 메세지 : ",messages)
    cursor = conn.cursor()
    sql = 'insert into chatbot(user_id, chat_num, message, chat_type, chat_time) values(%s, %s, %s, %s, %s)'
    for key,value in messages.items():
        cursor.execute(sql,(user_id, key, value[0],value[1], time))
    conn.commit()

@socketio.on('send_message')
def handle_message(data):
    print(data)
    return_data = None
    try:
        if data['BotType'] == "QUICK":
            with open("./json/quick_reply.json", "r", encoding='utf-8') as json_file:
                return_data = json.dumps(json.load(json_file))
            emit('receive_quick', {'response': f"{return_data}","Query": f"{data['Query']}"})
        elif data['BotType'] == "NORMAL":
            return_data = json.dumps(get_answer_from_engine(data['BotType'] ,data['Query']),ensure_ascii=False)
            emit('receive_normal', {'response': f"{return_data}"})
    except Exception as ex :
        print("Error : " ,ex)

@app.route('/save', methods=['POST'])
def save() :
    data = request.get_json()
    messages = data.get('messages')
    timestamp = datetime.strptime(data.get('timestamp'), "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
    user_id = data.get('user_id')
    data_dict = {}
    for idx, value in enumerate(messages):
        data_array = []
        if str(type(value['text'])) == "<class 'str'>":
            data_array.append(value['text'])
        else:
            text = ""
            for j in value['text']['props']['children']:
                if str(type(j)) == "<class 'str'>":
                    text += j
            data_array.append(text)
        data_array.append(value['type'])
        data_dict[idx] = data_array
    save_data(data_dict, timestamp, user_id)
    return jsonify({"data": "잘 받았슈"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)