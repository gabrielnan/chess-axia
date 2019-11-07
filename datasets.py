import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
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
        label = self.labels[i]
        bitboard = idxs_to_bitboard(idxs)
        inputs = defaultdict(lambda: [])
        counts = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
        for idx in idxs:
            if idx < RAW_BITBOARD_DIM:
                color, pos, piece = idx_to_piece(idx)
                inputs[PIECES[piece]].append(
                    torch.Tensor(append_pos(bitboard, pos)))
                counts[color][PIECES[piece]] += 1
        return inputs, counts, label

    def __len__(self):
        return len(self.all_idxs)


