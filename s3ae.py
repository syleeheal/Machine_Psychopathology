from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import gc


class S3AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim):
        super(S3AE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.label_dim = label_dim

    def forward(self, X):
        
        Z = self.encoder(X)
        Y_hat = (Z[:, :self.label_dim]) # sigmoid is applied in the loss function
        Z = Z.relu()
        X_hat = self.decoder(Z)

        return X_hat, Z, Y_hat
    

def infer_sae(sae, dataloader_test):
    sae.eval()
    with torch.no_grad(): 
        Xs, Ys, X_hats, Zs, y_hats = [], [], [], [], []
        for _X, _Y in dataloader_test:
            X_hat, Z, Y_hat = sae(_X.to(sae.encoder.weight.device))
            
            X_hats.append(X_hat.detach().cpu())
            Zs.append(Z.detach().cpu())
            y_hats.append(Y_hat.detach().cpu())
            Ys.append(_Y)
            Xs.append(_X)
            
        X_hat = torch.cat(X_hats, dim=0)
        Z = torch.cat(Zs, dim=0)
        Y_hat = torch.cat(y_hats, dim=0)
        Y = torch.cat(Ys, dim=0)
        X = torch.cat(Xs, dim=0)
        
    torch.cuda.empty_cache()
    gc.collect()
        
    return X, X_hat, Z, Y, Y_hat


