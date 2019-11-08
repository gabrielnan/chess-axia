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
        return 0
    return None


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
    net_wins = 0
    i = 0
    bar = tqdm(total=args.num_games)
    imbalance = max(args.num_games // 1000, 100)
    while i < args.num_games:
        if i % 1000 == 0:
            tqdm.write(f'# board positions: {len(idxs)}')

        game = read_game(games)
        label = get_label(game)
        if label is not None and abs(net_wins + (label * 2) - 1) < imbalance: 
            board = game.board()
            moves = list(game.mainline_moves())
            num_boards = min(args.num_samples, len(moves))
            move_idxs = set(np.random.choice(range(len(moves)), num_boards, replace=False))
            net_wins += (label * 2 - 1) * num_boards
            for j, move in enumerate(moves):
                board.push(move)
                if j in move_idxs:
                    idxs.append(get_idxs(board))
                    labels.append(label)
            i += 1
            bar.update(1)
    print(abs(len(labels) // 2 - sum(labels)))
    print(len(labels))
    bar.close()
    np.savez(args.boards_file, idxs=np.array(idxs), labels=np.array(labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-file', type=str, default='data/games.pgn')
    parser.add_argument('--num-games', type=int, default=800000)
    parser.add_argument('--boards-file', type=str, default='data/boards.npz')
    parser.add_argument('--num-samples', type=int, default=10)

    main(parser.parse_args())
