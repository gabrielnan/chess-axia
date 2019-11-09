import argparse
from comet_ml import Experiment

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AutoEncoder, BoardValuator
from utils import *
from datasets import BoardAndPieces

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Loading data')
    data = np.load(args.boards_file, allow_pickle=True)
    idxs = data['idxs']
    labels = data['labels']
    n = len(idxs)

    if args.shuffle:
        perm = np.random.permutation(n)
        idxs = idxs[perm]
        labels = labels[perm]

    experiment = Experiment(project_name="chess-axia")
    experiment.log_parameters(vars(args))

    print(f'Number of Boards: {n}')

    if torch.cuda.is_available() and args.num_gpus > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.num_train is None:
        args.num_train = n - args.num_test
    if args.num_train + args.num_test > n:
        raise ValueError('num-train and num-test sum to more than dataset size')
    train_idxs = idxs[:args.num_train]
    test_idxs = idxs[-args.num_test:]

    train_labels = labels[:-args.num_test]
    test_labels = labels[-args.num_test:]
    print(f'Win percentage: {sum(train_labels)/ len(train_labels):.1%}')
    print('Train size: ' + str(len(train_labels)))

    train_loader = DataLoader(BoardAndPieces(train_idxs, train_labels),
                              batch_size=args.batch_size, collate_fn=collate_fn,
                              shuffle=True)
    test_loader = DataLoader(BoardAndPieces(test_idxs, test_labels),
                             batch_size=args.batch_size, collate_fn=collate_fn)

    ae = AutoEncoder().to(device)
    ae_file = append_to_modelname(args.ae_model, args.ae_iter)
    ae.load_state_dict(torch.load(ae_file))

    model = BoardValuator(ae).to(device)
    if args.model_loadname:
        model.load_state_dict(torch.load(args.model_loadname))

    if args.ae_freeze:
        print('Freezing AE model')
        for param in ae.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    total_iters = 0

    for epoch in range(args.init_epoch, args.epochs):
        print(f'Running epoch {epoch} / {args.epochs}\n')
        #for batch_idx, (input, mask, label) in tqdm(enumerate(train_loader),
        #                             total=len(train_loader)):
        for batch_idx, (input, mask, label) in (enumerate(train_loader)):

            model.train()
            input = to(input, device)
            mask = to(mask, device)
            label = to(label, device)
            optimizer.zero_grad()
            loss, acc = model.loss(input, mask, label)
            loss.backward()

            train_losses.append(loss.item())
            optimizer.step()

            if total_iters % args.log_interval == 0:
                tqdm.write(f'Loss: {loss.item():.5f} \tAccuracy: {acc:.1%}')
                experiment.log_metric('accuracy', acc.item(), step=total_iters,
                                      epoch=epoch)
                experiment.log_metric('loss', loss.item(), step=total_iters,
                                      epoch=epoch)

            if total_iters % args.save_interval == 0:
                torch.save(model.state_dict(),
                           append_to_modelname(args.model_savename,
                                               total_iters))
                torch.save(model.state_dict(), args.model_savename)
                plot_losses(train_losses, 'vis/losses.png')

            if total_iters % args.eval_interval == 0 and total_iters != 0:
                loss, acc = eval(model, test_loader, device)
                tqdm.write(f'\tTEST: Loss: {loss:.5f} \tAccuracy: {acc:.1%}')
                test_losses.append(loss)
                experiment.log_metric('test accuracy', acc, step=total_iters,
                                      epoch=epoch)
                experiment.log_metric('test loss', loss, step=total_iters,
                                      epoch=epoch)
            total_iters += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boards-file', type=str, default='data/boards.npz')
    parser.add_argument('--ae-model', type=str, default='models/ae.pt')
    parser.add_argument('--ae-iter', type=int)
    parser.add_argument('--model-savename', type=str, default='models/axia.pt')
    parser.add_argument('--model-loadname', type=str)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--ae-freeze', action='store_true', default=False)
    parser.add_argument('--num-train', type=int)
    parser.add_argument('--num-test', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--init-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    main(parser.parse_args())
