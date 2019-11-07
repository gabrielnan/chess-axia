import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AutoEncoder
from utils import *
from datasets import Boards


def main(args):
    print('Loading data')
    idxs = np.load(args.boards_file, allow_pickle=True)['idxs']
    print(f'Number of Boards: {len(idxs)}')

    if torch.cuda.is_available() and args.num_gpus > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.shuffle:
        np.random.shuffle(idxs)

    train_idxs = idxs[:-args.num_test]
    test_idxs = idxs[-args.num_test:]

    train_loader = DataLoader(Boards(train_idxs), batch_size=args.batch_size,
                              shuffle=False)
    test_loader = DataLoader(Boards(test_idxs), batch_size=args.batch_size)

    model = AutoEncoder().to(device)
    if args.model_loadname:
        model.load_state_dict(torch.load(args.model_loadname))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    losses = []
    total_iters = 0

    for epoch in range(args.init_epoch, args.epochs):
        print(f'Running epoch {epoch} / {args.epochs}\n')
        for batch_idx, board in tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            board = board.to(device)
            optimizer.zero_grad()
            loss = model.loss(board)
            loss.backward()

            losses.append(loss.item())
            optimizer.step()


            if total_iters % args.log_interval == 0:
                tqdm.write(f'Loss: {loss.item()}')

            if total_iters % args.save_interval == 0:
                torch.save(model.state_dict(),
                           append_to_modelname(args.model_savename,
                                               total_iters))
                plot_losses(losses, 'vis/ae_losses.png')
            total_iters += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boards-file', type=str, default='data/boards.npz')
    parser.add_argument('--num-games', type=int, default=800000)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--num-test', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--init-epoch', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-savename', type=str, default='models/ae.pt')
    parser.add_argument('--model-loadname', type=str)

    main(parser.parse_args())
