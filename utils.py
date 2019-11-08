import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from torch._six import container_abcs, int_classes
import random

PIECES = ['p', 'n', 'b', 'r', 'q', 'k']
PIECE_IDX = {PIECES[i]: i for i in range(len(PIECES))}
RAW_BITBOARD_DIM = 64 * 6 * 2
BITBOARD_DIM = RAW_BITBOARD_DIM + 5
INPUT_DIM = BITBOARD_DIM + 64

class Bitboard(object):

    def __init__(self, board):
        self.bitboard, self.nonempty = self.get_bitboard(board)
        self.with_position = False


    @staticmethod
    def get_bitboard(board):
        nonempty = []
        bitboard = np.zeros(BITBOARD_DIM)
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
            idxs.append(pos + RAW_BITBOARD_DIM)
    return idxs


def idxs_to_bitboard(idxs):
    bitboard = np.zeros(BITBOARD_DIM)
    for idx in idxs:
        bitboard[idx] = 1
    return bitboard


def piece_to_idx(piece, pos):
    piece_idx = PIECE_IDX[piece.symbol().lower()]

    idx = piece.color * (64 * 6)
    idx += pos * 6
    idx += piece_idx
    return idx

def idx_to_piece(idx):
    if idx >= RAW_BITBOARD_DIM:
        raise ValueError('idx must be less than 64 * 6 * 2')
    color = idx // (64 * 6)
    idx %= 64 * 6
    pos = idx // 6
    piece = idx % 6
    return color, pos, piece

def idx_to_pos(idx):
    if idx >= RAW_BITBOARD_DIM:
        raise ValueError('idx must be less than 64 * 6 * 2')
    idx_no_color = idx % (64 * 6)
    return idx_no_color // 6

def append_pos(bitboard, pos):
    if len(bitboard) == BITBOARD_DIM:
        bitboard = np.append(bitboard, np.zeros(64))
    bitboard[BITBOARD_DIM + pos] = 1
    return bitboard

def sample_piece_pos(idxs):
    random_piece_idx = np.random.choice(idxs)
    while random_piece_idx >= RAW_BITBOARD_DIM:
        random_piece_idx = np.random.choice(idxs)
    return idx_to_pos(random_piece_idx)

def append_to_modelname(base, append):
    append = '' if append is None else append
    if base[-3:] == '.pt':
        base = base[:-3]
    return f'{base}{append}.pt'

def plot_losses(losses, savepath, title='Loss Curve'):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(savepath)


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    input_batch = defaultdict(lambda: [])
    mask_batch = defaultdict(lambda: [])
    for piece in PIECES:
        inputs = sum([elem[0][piece] for elem in batch], [])
        out = None
        elem = inputs[0]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x[0].numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        input_batch[piece] = torch.stack(inputs, 0, out=out)

        masks = [(color * 2 - 1) * collate_counts([elem[1][color][piece] for
                                                   elem in batch])
                 for color in range(2)]
        out = None
        elem = masks[0]
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in masks])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        mask_batch[piece] = torch.cat(masks, 1, out=out)

    label_batch = torch.Tensor([elem[2] for elem in batch])

    return input_batch, mask_batch, label_batch

def collate_counts(counts):
    mask = torch.zeros(len(counts), sum(counts))
    for i, count in enumerate(counts):
        for j in range(count):
            mask[i, j] = 1
    return mask

def to(obj, device):
    if isinstance(obj, dict):
        for key, val in obj.items():
            obj[key] = to(val, device)
        return obj

    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    raise ValueError(f'obj is neither tensor nor dict: {type(obj)}')

