from flask import Flask, render_template, g, redirect, url_for
from flask_socketio import SocketIO, emit
from models.model import Model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def startTraining():
	global model
	model = Model(socketio)
	socketio.start_background_task(trainModel)
	return render_template('model.html')

@app.route('/test')
def testModel():
	return render_template('test.html')

def trainModel():
	socketio.sleep(1)
	with app.open_resource('models/diabetes.csv') as f:
		model.trainModel(f)

@socketio.on('test')
def test(msg):
	model.testModel(msg['data'])

if __name__ == '__main__':
	socketio.run(app, debug=True)