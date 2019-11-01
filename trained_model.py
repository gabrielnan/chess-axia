from random import random

def trained_model(board_positions):
	"""
	trained_model: returns piece values based on board configuration
	Input: dictionary of board configuration with key=location, value=piece
	ex: {e4:wK, d5:bP}
	Output: dictionary of piece values with key=location, value=piece value
	ex: {e4:0.456881, d5:0.945984}
	"""
	
	piece_values = board_positions
	
	for key,_ in piece_values.items():
		piece_values[key] = random()
		
	return piece_values