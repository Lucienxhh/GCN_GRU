import argparse
import time
import numpy as np
import torch
import models
import torch.optim as optim    
from tqdm import tqdm
import torch.nn.functional as F
import utils

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--station', type=str, default='58027') 
parser.add_argument('--station_num', type=int, default=11)
parser.add_argument('--horizon', type=int, default=48)
parser.add_argument('--window', type=int, default=72)
parser.add_argument('--gru_layers', type=int, default=2)
parser.add_argument('--draw', type=bool, default=True)
parser.add_argument('--model', type=str, default='GCNGRU_Single', choices=['GCNGRU_Single', 'GCNGRU_Multi', 'GRU_Single', 'GRU_Multi'])
parser.add_argument('--batchify', type=bool, default=False, help='enable batchify during validating or testing') # if memory is not enough, set it True
parser.add_argument('--feature_dim', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--head_num', type=int, default=8, help='should be a factor of hidden_dim')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay rate')
parser.add_argument('--epoch_num', type=int, default=120)
args = parser.parse_args()

def test():
    model.eval()
    if args.batchify:
        pred_list = []
        for tx, ty in utils.get_batches(X=X_test, Y=y_test, batch_size=args.batch_size, shuffle=False):
            tx, ty = tx.to(args.device), ty.to(args.device)
            tx = tx.view(tx.shape[0], args.window, args.station_num, args.feature_dim) 
            with torch.no_grad():
                output = model(tx)
            pred_list.append(output)
        y_pred = torch.cat(pred_list, dim=0).cpu().numpy() * y_scale

    else:
        input = X_test.view(X_test.shape[0], args.window, args.station_num, args.feature_dim)
        with torch.no_grad():
            output = model(input.to(args.device))
        
        y_pred = output.reshape(-1, 1).cpu().numpy() * y_scale
    
    y_true = y_test.reshape(-1, 1).numpy() * y_scale
    utils.print_metrics(y_pred, y_true)

def validate():
    model.eval()
    total_loss = 0

    if args.batchify:  
        for tx, ty in utils.get_batches(X=X_valid, Y=y_valid, batch_size=args.batch_size, shuffle=False):
            tx, ty = tx.to(args.device), ty.to(args.device)
            tx = tx.view(tx.shape[0], args.window, args.station_num, args.feature_dim) 
            with torch.no_grad():
                output = model(tx)
                F.l1
        
            total_loss += F.mse_loss(output * y_scale, ty * y_scale)

    else:
        input = X_valid.view(X_valid.shape[0], args.window, args.station_num, args.feature_dim)
        with torch.no_grad():
            output = model(input.to(args.device))
        total_loss = F.mse_loss(output * y_scale, y_valid.to(args.device) * y_scale)
   
    
    return total_loss.item()

def train():
    model.train()

    for tx, ty in utils.get_batches(X=X_train, Y=y_train, batch_size=args.batch_size, shuffle=False):     
        optimizer.zero_grad()
        
        tx, ty = tx.to(args.device), ty.to(args.device)
        tx = tx.view(tx.shape[0], args.window, args.station_num, args.feature_dim)                  
        output = model(tx)
        loss = F.mse_loss(output * y_scale, ty * y_scale)                  
        loss.backward()
        optimizer.step()


Data = utils.DataLoader(file=args.station, horizon=args.horizon, window=args.window, missing_value=-9999) 

X_train, y_train, X_valid, y_valid, X_test, y_test = Data.X_train, Data.y_train, Data.X_valid, Data.y_valid, Data.X_test, Data.y_test

X_train, X_valid, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_valid), torch.FloatTensor(X_test)
y_train, y_valid, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_valid), torch.FloatTensor(y_test)
y_scale = Data.y_scale


t_train = time.time()
print('Start training...\n')

# Model and optimizer
if args.model == 'GCNGRU_Single':
    model = models.GCNGRU_Single(args.feature_dim, args.hidden_dim, args.dropout, args.horizon, args.gru_layers)
elif args.model == 'GCNGRU_Multi':
    model = models.GCNGRU_Multi(args.feature_dim, args.hidden_dim, args.dropout, args.horizon, args.gru_layers)
elif args.model == 'GRU_Single':
    args.hidden_dim = 64
    model = models.GRU_Single(args.feature_dim*args.station_num, args.hidden_dim, args.dropout, args.horizon, args.gru_layers)
else:
    args.hidden_dim = 64
    model = models.GRU_Multi(args.feature_dim*args.station_num, args.hidden_dim, args.dropout, args.horizon, args.gru_layers)

print("=== Tunable Parameters ===")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))
print()

model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

pbar = tqdm(range(args.epoch_num), total=args.epoch_num, ncols=100, unit="epoch", colour="red")

best_loss = np.inf
loss_values = []
for epoch in pbar:
    train()
    this_loss = validate()
    loss_values.append(this_loss)
    if this_loss < best_loss:
        torch.save(model.state_dict(), f'models/{args.station}_{args.horizon}_{args.model}.pt')
        best_loss = this_loss
    
    pbar.set_postfix({"current loss": this_loss, 'best loss in val': best_loss})
model.load_state_dict(torch.load(f'models/{args.station}_{args.horizon}_{args.model}.pt'))
print('Train time: {:.2f}'.format(time.time() - t_train))
test()
# if args.draw:
#     utils.draw_loss(loss_values, args.epoch_num)