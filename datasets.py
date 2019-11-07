import numpy as np
from torch.utils.data import Dataset
from utils import *


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


class BoardAndPieces(Dataset):
    def __init__(self, idxs, labels):
        self.all_idxs = idxs
        self.labels = labels

    def __getitem__(self, i):
        idxs = self.all_idxs[i]
        label = self.all_idxs[i]
        bitboard = idxs_to_bitboard(idxs)
        input = [{piece: [] for piece in PIECES},
                 {piece: [] for piece in PIECES}]
        for idx in idxs:
            color, pos, piece = idx_to_piece(idx)
            input[color][PIECES[piece]] = append_pos(bitboard, pos)

        return input, label

    def __len__(self):
        return len(self.all_idxs)


