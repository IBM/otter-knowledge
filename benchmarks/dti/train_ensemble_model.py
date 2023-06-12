import argparse

import pandas as pd
from benchmarks.dti.model import BindingAffinity, BindingAffinityInitial
import json
import torch
import numpy as np
import random
import os
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class DTIDataset(Dataset):
    def __init__(self, data_file, embeddings, gnn_embedding_dim, is_initial_embedding=False):
        self.embeddings = embeddings
        self.is_initial_embedding = is_initial_embedding
        self.gnn_embedding_dim = gnn_embedding_dim
        print("Read data from", data_file)
        df = pd.read_csv(data_file)
        print("Data shape", df.shape)
        self.data = [(r['Target'], r['Drug'], r['Y']) for index, r in df.iterrows()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        t, d, y = self.data[index]
        t = torch.tensor(self.embeddings['Target'][t], dtype=torch.float32)
        d = torch.tensor(self.embeddings['Drug'][d], dtype=torch.float32)
        if self.is_initial_embedding:
            t = t[self.gnn_embedding_dim:]
            d = d[self.gnn_embedding_dim:]
        return d, t, y


def merge_embeddings(xs, ys):
    rs = []
    for x, y in zip(xs, ys):
        r = {'Drug': x['Drug'], 'Target': x['Target']}
        for k, v in y['Drug'].items():
            r['Drug'][k] = v
        for k, v in y['Target'].items():
            r['Target'][k] = v
        rs.append(r)
    return rs


def get_embeddings(trains, tests):
    train_embeddings = []
    for train in trains:
        with open(train) as f:
            print("Load embeddings from", train)
            train_embedding = json.load(f)
            train_embeddings.append(train_embedding)
    test_embeddings = []
    for test in tests:
        with open(test) as f:
            print("Load embeddings from", test)
            test_embedding = json.load(f)
            test_embeddings.append(test_embedding)
    return merge_embeddings(train_embeddings, test_embeddings)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble training')
    parser.add_argument('--train',
                        default='./tdc/dti_dg/data/dti_dg_group/bindingdb_patent/train_val.csv',
                        type=str,
                        help='Root directory with the training data')
    parser.add_argument('--test',
                        default='./tdc/dti_dg/data/dti_dg_group/bindingdb_patent/test.csv',
                        type=str,
                        help='Root directory with the test data')
    parser.add_argument('--train_embeddings',
                        default='./tdc/dti_dg/data/train_val_dti_embeddings.json',
                        type=str,
                        help='Root directory with the embeddings of training drugs and proteins.')
    parser.add_argument('--test_embeddings',
                        default='./tdc/dti_dg/data/test_dti_embeddings.json',
                        type=str,
                        help='Root directory with the embeddings of test drugs and proteins.')
    parser.add_argument("--lr", default=5e-04, type=float,
                        help='Learning rate.')
    parser.add_argument("--steps", default=10000, type=int,
                        help='Maximum number of training steps')
    parser.add_argument("--seeds", default=0, type=int,
                        help='Random seeds.')
    parser.add_argument("--batch_size", default=256, type=int,
                        help='Mini batch size.')
    parser.add_argument("--is_initial_embeddings", default='no', type=str,
                        help='Set this value to yes if want to train with initial embeddings without GNN embeddings.')
    parser.add_argument("--gnn_embedding_dim", default=128, type=int,
                        help='Size of the GNN embeddings.')


    args = parser.parse_args()
    set_seed(args.seeds)
    with open(args.train_embeddings) as f:
        lines = f.readlines()
        train_embeddings = [l.rstrip("\n") for l in lines]
    with open(args.test_embeddings) as f:
        lines = f.readlines()
        test_embeddings = [l.rstrip("\n") for l in lines]

    combine_embeddings = get_embeddings(train_embeddings, test_embeddings)
    print(len(combine_embeddings))
    n_drug = len(list(combine_embeddings[0]['Drug'].values())[0])
    n_target = len(list(combine_embeddings[0]['Target'].values())[0])
    if args.is_initial_embeddings == 'yes':
        n_drug = n_drug - args.gnn_embedding_dim
        n_target = n_target - args.gnn_embedding_dim
    train_dataset = []
    test_dataset = []
    for i in range(len(combine_embeddings)):
        train_dataset.append(
            DTIDataset(args.train, combine_embeddings[i], is_initial_embedding=args.is_initial_embeddings == 'yes',
                       gnn_embedding_dim=args.gnn_embedding_dim))
        test_dataset.append(
            DTIDataset(args.test, combine_embeddings[i], is_initial_embedding=args.is_initial_embeddings == 'yes',
                       gnn_embedding_dim=args.gnn_embedding_dim))

    train_data = []
    test_data = []
    for i in range(len(combine_embeddings)):
        train_data.append(DataLoader(train_dataset[i], batch_size=args.batch_size, shuffle=True))
        test_data.append(DataLoader(test_dataset[i], batch_size=args.batch_size, shuffle=False))
    net = []
    for i in range(len(combine_embeddings)):
        if args.is_initial_embeddings == 'no':
            net.append(BindingAffinity(n_drug=n_drug,
                                       n_target=n_target,
                                       gnn_embedding_dim=args.gnn_embedding_dim))
        else:
            net.append(BindingAffinityInitial(n_drug=n_drug, n_target=n_target))
    print(args)

    print(net)
    optimizer = []
    for i in range(len(combine_embeddings)):
        optimizer.append(torch.optim.Adam(net[i].parameters(), lr=args.lr))
    moving_loss = 0.0
    n = 0
    run = True
    train_data_iter = [iter(data) for data in train_data]
    while run:
        for i in range(len(combine_embeddings)):
            try:
                data = next(train_data_iter[i])
            except:
                train_data_iter[i] = iter(train_data[i])
                data = next(train_data_iter[i])
            net[i].zero_grad()
            p = net[i](data[0].to(device), data[1].to(device))
            loss = torch.nn.MSELoss()(p.squeeze(-1), data[2].type(torch.float32).to(device))
            loss.backward()
            optimizer[i].step()
            moving_loss += loss.cpu().item()
            if n % 500 == 0:
                print("Step {}; Moving loss {}".format(n, moving_loss / (n + 1)))
        n += 1
        # test
        if n % 500 == 0:
            with torch.no_grad():
                ensemble_p = 0
                for i in range(len(combine_embeddings)):
                    all_p = []
                    all_label = []
                    pbar = tqdm(test_data[i])
                    for data in pbar:
                        p = net[i](data[0].to(device), data[1].to(device))
                        all_p.append(p.squeeze(-1))
                        all_label.append(data[2].type(torch.float32).to(device))
                    all_p = torch.cat(all_p)
                    ensemble_p += all_p
                    all_label = torch.cat(all_label)
                ensemble_p = ensemble_p/len(combine_embeddings)
                print(n, "Test pcc:", stats.pearsonr(ensemble_p.cpu().detach(), all_label.cpu().detach())[0])

        if n == args.steps:
            run = False
            break
