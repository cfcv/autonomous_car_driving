import socketio
from flask import Flask
import eventlet
import eventlet.wsgi
from io import BytesIO
import base64
from PIL import Image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import training_utils as TU

server = socketio.Server()

app = Flask(__name__)

@server.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0,0)

def send_control(steering_angle, throttle):
    server.emit("steer", 
                data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()},
                skip_sid=True)

MAX_SPEED = 25
MIN_SPEED = 10

@server.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        image = np.asarray(image)
        image = TU.pre_process(image)
        image = np.array([image]) #get a 4D tensor
        
        
        if(speed > MAX_SPEED):
            limit = MIN_SPEED
        else:
            limit = MAX_SPEED

        steering_angle = float(model.predict(image, batch_size=1))
        throttle = np.clip(1.0 - abs(steering_angle) - (speed/limit)**2, -1.0, 1.0)
        
        #------------------ only for track 2 ------------------
        steering_angle *= 1.5

        throttle = 1.0 - steering_angle**2 - (speed/limit)**2
        if(speed < 0.4):
            throttle = - throttle 
        elif(speed < 4):
            print('ENTROU')
            throttle = min(throttle + 0.3, 0.9)
        #------------------

        print('{} {} {}'.format(steering_angle, throttle, speed))
        
        send_control(steering_angle, throttle)
    else:
        server.emit("manual", data={}, skip_sid=True)

if __name__ == '__main__':
    model = load_model("track2_30_epochs_BN.h5")
    app = socketio.Middleware(server, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
