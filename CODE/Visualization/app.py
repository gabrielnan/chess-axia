from flask import Flask, render_template, request, jsonify
from os import path
import chess as pychess
from json import loads
from play import Player
import sys

# Parse arguments
numArguments = len(sys.argv)
if numArguments > 2:
	print("\nUSAGE:")
	print("python app.py [port]")
	print('\nOptionally provide one argument, which is the local port to serve the visualization to.  Default is 5001.\n')
	raise ValueError('Invalid number of arguments.  Use 0 or 1 argument.')
elif numArguments == 2:
	port = sys.argv[1]
elif numArguments == 1:
	port = 5001
	
app = Flask(__name__)
app.secret_key = 's3cr3t'
app.debug = True
app._static_folder = path.abspath("templates/static/")
						   
@app.route('/', methods=['GET'])
def chess():
    title = 'CSE6242 Team 44 Chess App'
    return render_template('layouts/chess.html', title=title)

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    board_positions = request.form.to_dict()
    print('INPUT:', board_positions)
    board_positions = list(board_positions.keys())[0]
    board = pychess.Board(board_positions)
    piece_values = player.get_values(board)
    print('OUTPUT: ', piece_values)
    return jsonify(piece_values)

if __name__ == '__main__':
    model_path = 'models/axia_15_626000.pt'
    player = Player(model_path)
    app.run(host='0.0.0.0', port=port)
