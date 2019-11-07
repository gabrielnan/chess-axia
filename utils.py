import numpy as np
from torch.utils.data import Dataset
import torch
import random

PIECES = ['p', 'n', 'b', 'r', 'q', 'k']
PIECE_IDX = {PIECES[i]: i for i in range(len(PIECES))}

class Bitboard(object):

    def __init__(self, board):
        self.bitboard, self.nonempty = self.get_bitboard(board)
        self.with_position = False


    @staticmethod
    def get_bitboard(board):
        nonempty = []
        bitboard = np.zeros(64*6*2 + 5)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                nonempty.append(i)
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
        return bitboard, nonempty

def get_idxs(board):
    idxs = []
    for pos in range(64):
        piece = board.piece_at(pos)
        if piece:
            idx = piece_to_idx(piece, pos)
            idxs.append(idx)

    meta = [
        board.turn,
        board.has_kingside_castling_rights(True),
        board.has_kingside_castling_rights(False),
        board.has_queenside_castling_rights(True),
        board.has_queenside_castling_rights(False),
    ]
    for pos in range(5):
        if meta[pos]:
            idxs.append(pos + 64 * 6 * 2)
    return idxs


def idxs_to_bitboard(idxs):
    bitboard = np.zeros(64*6*2 + 5 + 64)
    for idx in idxs:
        bitboard[idx] = 1
    return bitboard

def piece_to_idx(piece, pos):
    piece_idx = PIECE_IDX[piece.symbol().lower()]

    idx = piece.color * (64 * 6)
    idx += pos * 6
    idx += piece_idx
    return idx

def idx_to_pos(idx):
    if idx >= 64 * 6 * 2:
        raise ValueError('idx must be less than 64 * 6 * 2')
    idx_no_color = idx % (64 * 6)
    return idx_no_color // 6

def append_pos(bitboard, pos):
    if len(bitboard) == 64 * 6 * 2 + 5:
        bitboard = np.append(bitboard, np.zeros(64))
    bitboard[64 * 6 * 2 + 5 + pos] = 1
    return bitboard

def sample_piece_pos(idxs):
    random_piece_idx = np.random.choice(idxs)
    while random_piece_idx >= 64 * 6 * 2:
        random_piece_idx = np.random.choice(idxs)
    return idx_to_pos(random_piece_idx)

class Boards(Dataset):
    def __init__(self, idxs):
        self.all_idxs = idxs

    def __getitem__(self, i):
        idxs = self.all_idxs[i]
        bitboard = idxs_to_bitboard(idxs)

        # fix bug of [-5] should consider all numbers under 64 * 6 * 2
        random_piece_pos = sample_piece_pos(idxs)
        bitboard_with_pos = append_pos(bitboard, random_piece_pos)
        result = torch.Tensor(bitboard_with_pos)
        return result

    def __len__(self):
        return len(self.all_idxs)

def append_to_modelname(base, append):
    if base[-3:] == '.pt':
        base = base[:-3]
    return f'{base}{append}.pt'
