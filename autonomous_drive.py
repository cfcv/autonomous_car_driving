import socketio
from flask import Flask

server = socketio.Server()

app = Flask(__name__)