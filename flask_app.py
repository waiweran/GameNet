from flask import Flask, request, jsonify
import numpy as np
import json

import gain, q_learn, adversarial_train

global model
global agent
global agent_pre
global raw
global good
global agents

app = Flask(__name__)
model = gain.GainModel()
agent = q_learn.GainDQN(file='checkpoints/gain-dqn.h5', epsilon=0.0)
agent_pre = q_learn.GainDQN(file='checkpoints/gain-pretrain.h5', epsilon=0.0)
raw = adversarial_train.GainNet(file='checkpoints/adversarial_raw.h5')
good = adversarial_train.GainNet(file='checkpoints/adversarial_good.h5')

agents = dict()

@app.route('/predict/')
def predict():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            output = model.run_game(instring)
            return jsonify(output)
    return jsonify(['Incorrect Input'])

@app.route('/predict/raw/')
def predict_raw():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            indict = json.loads(instring)
            netin = np.expand_dims(indict['GainChoice'], 0)
            output = raw.predict(netin)
            return jsonify(output.tolist())
    return jsonify(['Incorrect Input'])

@app.route('/predict/good/')
def predict_good():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            indict = json.loads(instring)
            netin = np.expand_dims(indict['GainChoice'], 0)
            output = good.predict(netin)
            return jsonify(output.tolist())
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

@app.route('/predict/pretrain/')
def predict_pretrain():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            indict = json.loads(instring)
            netin = np.expand_dims(indict['GainChoice'], 0)
            output = agent_pre.predict(netin)
            return jsonify(output.tolist())
    return jsonify(['Incorrect Input'])

@app.route('/predict/dqn/<netname>/')
def predict_specific(netname):
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            indict = json.loads(instring)
            netin = np.expand_dims(indict['GainChoice'], 0)
            new_agent = q_learn.GainDQN(file=('checkpoints/{}.h5'.format(netname)))
            output = new_agent.predict(netin)
            return jsonify(output)
    return jsonify(['Incorrect Input'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
