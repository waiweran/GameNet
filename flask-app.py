from flask import Flask, request, jsonify
import gain

global model

app = Flask(__name__)
model = gain.GainModel()

models = dict()

@app.route('/predict/')
def predict():
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            output = model.run_game(instring)
            return jsonify(output)
    return jsonify(['Incorrect Input'])

@app.route('/predict/<netname>/')
def predict_specific(netname):
    if netname not in models:
        models[netname] = gain.GainModel(netname=netname)
    if 'datain' in request.args:
        instring = request.args['datain']
        if instring.startswith('{'):
            output = models[netname].run_game(instring)
            return jsonify(output)
    return jsonify(['Incorrect Input'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
