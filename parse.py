import numpy as np
import argparse
from tqdm import tqdm, trange
from chess.pgn import read_game
from utils import get_idxs

PIECE_IDX = {
    'p': 0,
    'n': 1,
    'b': 2,
    'r': 3,
    'q': 4,
    'k': 5,
}

def get_label(game):
    result = game.headers['Result']
    result = result.split('-')
    if result[0] == '1':
        return 1
    elif result[0] == '0':
        return -1
    else:
        return 0


def get_bitboard(board):
    bitboard = np.zeros(64*6*2 + 5)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            color = int(piece.color) + 1
            piece_idx = PIECE_IDX[piece.symbol().lower()] + i * 6
            bitboard[piece_idx * color] = 1
    bitboard[-5:] = [
        board.turn,
        board.has_kingside_castling_rights(True),
        board.has_kingside_castling_rights(False),
        board.has_queenside_castling_rights(True),
        board.has_queenside_castling_rights(False),
    ]
    return bitboard


def main(args):
    games = open(args.games_file)
    idxs = []
    labels = []
    for i in trange(args.num_games):
        if i % 1000 == 0:
            tqdm.write(f'# board positions: {len(idxs)}')

        game = read_game(games)
        label = get_label(game)
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            idxs.append(get_idxs(board))
            labels.append(label)
    np.savez(args.boards_file, idxs=np.array(idxs), labels=np.array(labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-file', type=str, default='data/games.pgn')
    parser.add_argument('--num-games', type=int, default=800000)
    parser.add_argument('--boards-file', type=str, default='data/boards.npz')

    main(parser.parse_args())