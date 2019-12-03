import argparse
from chess.pgn import read_game

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AutoEncoder, BoardValuator
from utils import *
from datasets import BoardAndPieces


games_file = 'data/games.pgn'
games = open(games_file)
game = read_game(games)
board = game.board()
moves = list(game.mainline_moves())
for move in moves[:15]:
    board.push(move)

print(board)
idxs = get_idxs(board)
bitboard = idxs_to_bitboard(idxs)
inputs = defaultdict(lambda: [])
counts = defaultdict(lambda: [0, 0])
positions = defaultdict(lambda: [])
for idx in idxs:
    if idx < RAW_BITBOARD_DIM:
        color, pos, piece = idx_to_piece(idx)
        input = torch.Tensor(append_pos(bitboard, pos))
        if color == 0:
            inputs[PIECES[piece]].insert(0, input)
            positions[PIECES[piece]].insert(0, pos_to_str(pos))
        else:
            inputs[PIECES[piece]].append(input)
            positions[PIECES[piece]].append(pos_to_str(pos))
        counts[PIECES[piece]][color] += 1

in_batch, mask_batch, _ = collate_fn([(inputs, counts, 0)])

#########33

# ae = AutoEncoder()
# ae_file = append_to_modelname(args.ae_model, args.ae_iter)
# ae.load_state_dict(torch.load(ae_file))

ae = AutoEncoder()
model = BoardValuator(ae)
model_loadname = 'models/axia_5_242000.pt'

if model_loadname:
    model.load_state_dict(torch.load(model_loadname,
                                     map_location=torch.device('cpu')))

model.get_inputs()
out_batch = model(in_batch, mask_batch)
out_batch
key = 'q'
print(model.models[key](in_batch[key]).std())
i = 1


for one, two in zip([1, 2], a):
    print(one, two.item())

