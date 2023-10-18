import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


# 多步预测
class GCNGRU_Multi(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, horizon, gru_layers):
        super(GCNGRU_Multi, self).__init__()

        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.gru_layers = 2
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=gru_layers)
        self.fc = nn.Linear(hidden_dim, 1)  

    def forward(self, features):
        #batch_size, window_size, station_num, feature_dim
        b, w, s, f = features.shape
        h = self.hidden_dim

        node = features.view(-1, f)  # (b,w,s,f) -> (b*w*s,f)

        edge = torch.LongTensor([(s*i, i*s+j) for i in range(b*w) for j in range(1,s)]).t().to(node.device)
        
        x = self.conv1(node, edge)
        x = self.conv2(x, edge)

        restored_x = x.view(b, w, s, h)
        
        # GRU
        x = restored_x[:,:,0,:]
        hi = torch.zeros(self.gru_layers, x.size(0), h).to(x.device)
        output = []
        for _ in range(self.horizon):
            x, hi = self.gru(x, hi)
            output.append(hi[-1,:,:])

        output = torch.stack(output)
        output = self.fc(output)
        output = output.squeeze().t()
        return output

# 单步预测
class GCNGRU_Single(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, horizon, gru_layers):
        super(GCNGRU_Single, self).__init__()       
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=gru_layers)
        self.fc = nn.Linear(hidden_dim, horizon)  

    def forward(self, features):
        #batch_size, window_size, station_num, feature_dim
        b, w, s, f = features.shape
        h = self.hidden_dim

        node = features.view(-1, f)  # (b,w,s,f) -> (b*w*s,f)

        edge = torch.LongTensor([(s*i, i*s+j) for i in range(b*w) for j in range(1,s)]).t().to(node.device)
        
        x = self.conv1(node, edge)
        x = self.conv2(x, edge)
        restored_x = x.view(b, w, s, h)
        
        # GRU
        x = restored_x[:,:,0,:]
        _, hidden = self.gru(x)

        return self.fc(hidden[-1,:,:])


# 单步预测
class GRU_Single(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, horizon, gru_layers):
        super(GRU_Single, self).__init__()
        
        self.gru_layers = gru_layers
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=gru_layers)
        self.fc = nn.Linear(hidden_dim, horizon)  

    def forward(self, features):
        #batch_size, window_size, station_num, feature_dim
        b, w, s, f = features.shape

        x = features.view(b, w, s*f)  # (b,w,s,f) -> (b,w,s*f)

        _, hidden = self.gru(x)

        return self.fc(hidden[-1,:,:])


# 多步预测
class GRU_Multi(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, horizon, gru_layers):
        super(GRU_Multi, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.horizon = horizon
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=gru_layers)
        self.fc = nn.Linear(hidden_dim, 1)  

    def forward(self, features):
        #batch_size, window_size, station_num, feature_dim
        b, w, s, f = features.shape

        x = features.view(b, w, s*f)  # (b,w,s,f) -> (b,w,s*f)
        x = self.linear(x)

        hi = torch.zeros(self.gru_layers, x.size(0), self.hidden_dim).to(x.device)
        output = []
        for _ in range(self.horizon):
            x, hi = self.gru(x, hi)
            output.append(hi[-1,:,:])

        output = torch.stack(output)
        output = self.fc(output)
        output = output.squeeze().t()

        return output