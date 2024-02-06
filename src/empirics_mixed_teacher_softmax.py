import numpy as np
import torch
from math import *
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
import time

from src.config import RESULT_DIR
from utils.experiments import dump_pickle, load_pickle

result_dir = RESULT_DIR / 'empirics'
result_dir.mkdir(exist_ok=True, parents=True)

device =  "cuda:0"#"cuda:0"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_ground_truth(d, L, r, delta=1.,which_A='standard'):
    if delta is None:
        delta = 0.0

    center = torch.ones(d)
    mu = torch.vstack((center, -center))

    Q_star = torch.randn(d) 
    #A = torch.Tensor(np.fliplr(np.eye(2)).copy())
    if which_A=='standard':
        A = torch.Tensor(np.array([[.6,.4],[.4,.6]]))
    elif which_A=='2ndexample':
        A = torch.Tensor(np.array( [[0.3, 0.7 ],[0.8,0.2]]))
    else: 
        raise ValueError("which_A not recognized")

    return center, mu, Q_star, A


def get_data(n, mu, sigma, Q_star, d, L, r, A,  omega=None):
    mus = np.repeat(mu[np.newaxis, :, :], n, axis=0)/np.sqrt(d)
    noises = sigma*torch.randn((n, L, d))
    
    # softmax
    sftm=torch.nn.Softmax(dim=-1)

    # compute y
    xQ = torch.einsum("imk,k->im", noises, Q_star/np.sqrt(d))
    xQKx = sftm(torch.einsum("im,in->imn", xQ, xQ))
    y = (1-omega) * torch.einsum("imn,inj->imj", xQKx, noises) + omega * torch.einsum("nld,fl->nfd", noises, A)

    return mus+noises, y, noises


class VectorDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X.to(device), Y.to(device)
        self.samples = X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.samples


class LinearModel(torch.nn.Module):  # linear attention
    def __init__(self, L):
        super(LinearModel, self).__init__()
        self.B = torch.nn.Parameter(torch.randn(L, L))
        # self.K=torch.nn.Parameter(torch.randn(d,r))

    def forward(self, x):
        yhat = self.B@x
        return yhat


class AttentionModel(torch.nn.Module):
    def __init__(self, Q_star, d, r, informed=False,informed_position=False):
        super(AttentionModel, self).__init__()
        self.d = d
        self.activ = torch.nn.Softmax(dim=-1)
        
        if informed and informed_position:
            raise ValueError("informed and informed_position cannot be both true")

        # Warm start
        if informed:
            self.Q = torch.nn.Parameter(Q_star.reshape(-1, 1))
        elif informed_position:
            self.Q = torch.nn.Parameter(torch.ones(d).reshape(-1, 1)/np.sqrt(d))
        else:
            self.Q = torch.nn.Parameter(0.1*torch.randn(d, r))
            
        

    def forward(self, x):
        xQ = torch.einsum("imk,kl->iml", x, self.Q/np.sqrt(self.d))
        # todo why not K?
        xQKx = self.activ(torch.einsum("iml,inl->imn", xQ, xQ))
        yhat = torch.einsum("imn,inj->imj", xQKx, x)
        return yhat


def quadloss(ypred, y):
    d = y.shape[-1]
    return torch.sum((ypred-y)**2)/2/d


def train_attention(train_loader, X_test, y_test, mu, center, Q_star, d, L, r, lmbda, tol=1e-5, verbose=False, tied=False, informed=False,optim='adam',informed_position=False):
    model = AttentionModel(Q_star, d, r,informed=informed,informed_position=informed_position).to(device)
    
    if optim == 'adam':

        optimizer = torch.optim.Adam(
            [{'params': [model.Q],"weight_decay":lmbda }], lr=0.01) #5e-1 #
        n_iter = 2500
    elif optim == 'GD':
        optimizer = torch.optim.SGD(
            [{'params': [model.Q],"weight_decay":lmbda }], lr=0.15) #5e-1 #
        n_iter = 5000
    else:
        raise ValueError("optim not recognized")
    
    gen_Loss_list = []
    Loss_list = []
    mq = []
    thet = []
    
    

    # Record the start time
    start_time = time.time()


    for t in range(n_iter):
        if t % 500 == 0:
            print(t)
        for x, y in train_loader:

            y_pred = model(x)

            loss = quadloss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_list.append(loss.item()+lmbda/2*float(torch.sum(model.Q.cpu().flatten()**2)))
        if t > n_iter - 20:
            
            gen_Loss_list.append(quadloss(model(X_test),y_test).item()/y_test.shape[0])

            Q = model.Q.cpu().flatten()

            magq = np.abs(float(center@Q/d))

            mq.append(magq)
            thet.append(np.abs(float(Q_star@Q/d)))

    # xQ=torch.einsum("imk,k->im",X_test.detach().cpu(),Q/np.sqrt(d))
    # Record the end time
    end_time = time.time()
    import matplotlib.pyplot as plt
    plt.plot(Loss_list)
    plt.show()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(elapsed_time)
    # true theta from notes
    return np.mean(gen_Loss_list[-5:]), np.mean(mq[-5:]), np.mean(thet[-5:]), np.mean(Loss_list[-5:]), Loss_list


def train_linear(train_loader, X_test, y_test, mu, Q_star, d, L, r, lmbda, tol=1e-5, verbose=False, tied=False):

    model = LinearModel(L).to(device)

    optimizer = torch.optim.Adam(
        [{'params': [model.B], "weight_decay":lmbda}], lr=5e-1)

    gen_Loss_list = []
    Loss_list = []
    n_iter = 200

    for t in range(n_iter):
        if t % 500 == 0:
            print(t)
        for x, y in train_loader:

            y_pred = model(x)

            loss = quadloss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t > n_iter - 20:
            Loss_list.append(loss.item()+lmbda/2*float(torch.sum(model.B.cpu().flatten()**2)))
            gen_Loss_list.append(quadloss(model(X_test),y_test).item()/y_test.shape[0])
            
        print(loss.item())

    return np.mean(gen_Loss_list[-5:]), np.mean(Loss_list[-5:])


def get_errors(alphas, d, L, r, sigma=1., N_iter=5, delta=1.0, omega=None, lmbda_att=0.01, lmbda_linear=0.001, informed=False,optim='GD',informed_position=False,which_A='standard'):

    results = []

    for alpha in alphas:
        n = int(np.round(alpha*d))

        eg = []  # gen error
        er = [] # train error
        m = []  # magnetization m
        lin = []  # error of linear model
        t = []  # teacher/student overlap theta
        loli = []

        for j in range(N_iter):

            center, mu, Q_star, A = make_ground_truth(d, L, r, delta,which_A)

            X_test, y_test, X_test_linear = get_data(50000, mu, sigma, Q_star, d, L, r, A, omega)
            X_train, y_train, X_train_linear  = get_data(n, mu, sigma, Q_star, d, L, r, A, omega)
            y_test = y_test.to(device)
            X_test = X_test.to(device)
            X_test_linear = X_test_linear.to(device)
            dataset_att = VectorDataset(X_train, y_train)

            dataset_linear = VectorDataset(X_train_linear, y_train)

            # put batch size n if one wants to do GD
            train_loader_att = DataLoader(dataset_att, batch_size=int(n))
            train_loader_linear = DataLoader(dataset_linear, batch_size=int(n))

            e, mag, th, e_train, loss_list = train_attention(train_loader_att, X_test, y_test, mu,
                                         center, Q_star, d, L, r, lmbda=lmbda_att, informed=informed,optim=optim,informed_position=informed_position)
            elin, elin_train = train_linear(train_loader_linear, X_test_linear, y_test,
                                mu, Q_star, d, L, r, lmbda=lmbda_linear)

            print("mag", mag, "error ", e, "error_lin ", elin)
            eg.append(e)
            er.append(e_train)
            m.append(mag)
            t.append(th)
            lin.append(elin)
            loli.append(loss_list)
            
            del X_test, y_test, X_test_linear
            del X_train, y_train, X_train_linear

        eg = np.array(eg)
        t = np.array(t)
        m = np.array(m)
        lin = np.array(lin)
        er = np.array(er)

        results.append({
            'alpha': alpha,
            'omega': omega,
            'sigma': sigma,
            'delta': delta,
            'd': d,
            'L': L,
            'r': r,
            'N_iter': N_iter,
            'attention_gen_error': eg,
            'attention_theta': t,
            'attention_magnetization': m,
            'linear_gen_error': lin,
            'linear_lmbda': lmbda_linear,
            'attention_lmbda': lmbda_att,
            'attention_train_error': er,
            'attention_train_error_curves':loli,
            'informed': informed,
            'informed_position': informed_position,
            'optim': optim,
            'which_A':which_A
        })

    return pd.DataFrame(results)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your program description here')
    parser.add_argument('--alphas', type=float, nargs=3, default=[0.1, 3, 10])
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--L', type=int, default=2)
    parser.add_argument('--r', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.5)
    parser.add_argument('--omega', type=float, default=0.2)
    parser.add_argument('--lmbda', type=float, default=0.01)
    parser.add_argument('--lmbda_linear', type=float, default=0.0001)
    parser.add_argument('--N_iter', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--informed', action='store_true') #whether to use informed init or not
    parser.add_argument('--optim', type =str,default='GD')
    parser.add_argument('--informed_position', action='store_true') #whether to use informed init or not
    parser.add_argument('--which_A', type =str,default='standard')
    args = parser.parse_args()

    # Extract the three alphas values
    alpha_start, alpha_end, alpha_count = args.alphas
    alphas = np.linspace(alpha_start, alpha_end, int(alpha_count))
    args.alphas = alphas

    return args

if __name__ == '__main__':
    args = parse_arguments()

    # Construct the filename using the parsed arguments
    if args.exp_name is not None:
        result_dir = result_dir / args.exp_name
        result_dir.mkdir(exist_ok=True, parents=True)
        
    seed = np.random.rand() * 10000000

    filename = result_dir / f"d={args.d}.sigma={args.sigma}.delta={args.delta}.omega={args.omega}.lmbda={args.lmbda}.lmbda_linear={args.lmbda_linear}.informed={args.informed}.optim={args.optim}.{seed}.pkl"

    # Use the parsed arguments and generated alphas in your function call
    results = get_errors(args.alphas, d=args.d, L=args.L, r=args.r, sigma=args.sigma, delta=args.delta,
                         omega=args.omega, lmbda_att=args.lmbda, lmbda_linear=args.lmbda_linear, N_iter=args.N_iter,
                         informed=args.informed,optim=args.optim,informed_position=args.informed_position,which_A=args.which_A)
    try:
        df = load_pickle(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    dump_pickle(pd.concat([df, results], axis=0), filename)

