from flask import Flask, render_template, request, jsonify
from os import path
from json import loads
from trained_model import trained_model

app = Flask(__name__)
app.secret_key = 's3cr3t'
app.debug = True
app._static_folder = path.abspath("templates/static/")
						   
@app.route('/', methods=['GET'])
def chess():
    title = 'Chess'
    return render_template('layouts/chess.html', title=title)

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    board_positions = request.form.to_dict()
    print('INPUT DICT:',board_positions)
    piece_values = trained_model(board_positions)
    print('OUTPUT DICT: ', piece_values)
    return jsonify(piece_values)

def calculate_values(board_positions):
    piece_values = board_positions
    for key,_ in piece_values.items():
        piece_values[key] = random()
    return piece_values

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
