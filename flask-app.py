from flask import Flask, request, jsonify
import gain, q_learn
import json

global model
global agent
global models

app = Flask(__name__)
model = gain.GainModel()
agent = q_learn.GainDQN(file='checkpoints/gain-dqn.h5')
agent.epsilon = 0.0

models = dict()

@app.route('/predict/')
def predict():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            output = model.run_game(instring)
            return jsonify(output)
    return jsonify(['Incorrect Input'])

@app.route('/predict/dqn/')
def predict_dqn():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            indict = json.loads(instring)
            netin = np.expand_dims(indict['GainChoice'], 0)
            output = agent.predict(netin)
            return jsonify(output.tolist())
    return jsonify(['Incorrect Input'])

@app.route('/predict/<netname>/')
def predict_specific(netname):
    if netname not in models:
    	models[netname] = gain.GainModel(netname)
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            output = models[netname].run_game(instring)
            return jsonify(output)
    return jsonify(['Incorrect Input'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
