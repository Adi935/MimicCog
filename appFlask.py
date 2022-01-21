from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from PIL import Image

from sklearn.feature_extraction import image
from detectObjectMy import *

from flask_socketio import SocketIO

app = Flask(__name__)
# sockets = Sockets(app)
socketio = SocketIO(app)

HTTP_SERVER_PORT = 5000

@app.route('/')
def home():
	return render_template('./index.html')

@app.route('/enter')
def secondPage():
	return render_template('./secondPageWebcam.html')

@socketio.on('connect')
def func():
    print("Hello!")

@socketio.on('getFrames')
def handleMessage(image_data_url):
    img = base64.b64decode(image_data_url)
    b = io.BytesIO(img)
    pimg = Image.open(b)
    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    if detectObj(frame) is not None:
        final = detectObj(frame)
        print(final)
        socketio.emit('getOutput',final)
    # detectObj(img)

if __name__ == '__main__':
    # app.logger.setLevel(logging.DEBUG)
    # from gevent import pywsgi
    # from geventwebsocket.handler import WebSocketHandler

    socketio.run(app)
    # server = pywsgi.WSGIServer(('', HTTP_SERVER_PORT), app, handler_class=WebSocketHandler)
    # print("Server listening on: http://localhost:" + str(HTTP_SERVER_PORT))
    # server.serve_forever()


# ############################################## THE REAL DEAL ###############################################
# @app.route('/detectObject' , methods=['POST'])
# def mask_image():
# 	# print(request.files , file=sys.stderr)
# 	file = request.files['image'].read() ## byte file #here it reads a image as a byte file, we should read a video
# 	npimg = np.fromstring(file, np.uint8)
# 	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
# 	######### Do preprocessing here ################
# 	# img[img > 150] = 0
# 	## any random stuff do here
# 	################################################

# 	num = detectObj(img)		##here we should replace with our function
# 	## so we need a variable class_no = run_model(img)
# 	return jsonify({'status':str(num)}) ##We can directly return switcher function ka output

##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

# @app.route('/test' , methods=['GET','POST'])
# def test():
# 	print("log: got at test" , file=sys.stderr)
# 	return jsonify({'status':'succces'})



	
# @app.after_request
# def after_request(response):
#     print("log: setting cors" , file = sys.stderr)
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response


# if __name__ == '__main__':
# 	app.run(debug = True)
