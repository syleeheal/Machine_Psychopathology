from tqdm import tqdm, trange
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
    

def train_sae(sae, optimizer, dataloader_train, dataloader_test, num_epochs, run, grad_norm=5, label_weight=None, recon_weight=1, cls_weight=1):
    
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.BCEWithLogitsLoss(weight=label_weight)

    sample_size = len(dataloader_train.dataset)
    scheduler_period = 30
    patience_thre =  60

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_period,T_mult=1,eta_min=1e-5,)
    
    device = sae.encoder.weight.device
    iterator = trange(num_epochs, desc='Cls. Loss: ', leave=False)
    best_loss = float('inf')
    patience = 0
    
    for epoch in iterator:
            
        train_epoch_loss = 0
        recon_loss_epoch = 0
        sparsity_loss_epoch = 0
        class_loss_epoch = 0

        for X, Y in dataloader_train:
                        
            output, z, y_pred = sae(X)
            
            l1_norm = sae.decoder.weight.norm(dim=0, p=1)
            
            class_loss = criterion_2(y_pred, Y) * cls_weight
            sparsity_loss = (z*l1_norm).abs().mean() * 0.001
            recon_loss = criterion_1(output, X) * recon_weight

            train_loss = (recon_loss) + (sparsity_loss) + (class_loss)
                
            train_epoch_loss += train_loss.item()
            recon_loss_epoch += recon_loss.item()
            sparsity_loss_epoch += sparsity_loss.item()
            class_loss_epoch += class_loss.item()
            
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_norm)
            optimizer.step()
            
            iterator.set_description('Cls. Loss: {:.4f}'.format(class_loss.item()))

        recon_loss_epoch /= (sample_size // batch_size)
        sparsity_loss_epoch /= (sample_size // batch_size)
        class_loss_epoch /= (sample_size // batch_size)
        if run is not None:
            run.log({'Recon Loss': recon_loss_epoch, 'Sparsity Loss': sparsity_loss_epoch, 'Class Loss': class_loss_epoch})

        patience += 1
        if (recon_loss_epoch + class_loss_epoch) < best_loss:
            best_loss = (recon_loss_epoch + class_loss_epoch)
            best_s3ae = sae
            patience = 0
        if patience == patience_thre:
            print(f'Early stopping at epoch {epoch+1}.')
            break

        scheduler.step()

    print('Reconstruction Loss:', recon_loss.item(), 'Sparsity Loss:', sparsity_loss.item(), 'Class Loss:', class_loss.item())
    print('Training complete.')
        
    return best_s3ae, optimizer, best_loss


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
                
    return X, X_hat, Z, Y, Y_hat


