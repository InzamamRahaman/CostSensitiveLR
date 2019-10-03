import loss
import nn
import torch as th
import torch.optim as optim
import torch.utils.data as data_utils
import dataset


def train_cost_sensitive_lr(X, y, num_inputs, b00, b01, b10, b11, epochs, lr=0.01, shuffle=True,
                            batch_size=100, use_cuda=True, lam=0.0005):
    dset = dataset.PairedDataset(X, y)
    loader = data_utils.DataLoader(dataset=dset, batch_size=batch_size, shuffle=shuffle)
    model = nn.CostSensitiveLRLayer(num_inputs)
    use_cuda = use_cuda and th.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adagrad(model.parameters(), lr)
    loss_curve = []
    for i in range(epochs):
        total_loss = 0
        for bn, (X_i, y_i) in enumerate(loader):
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            model.zero_grad()
            model_output = model(X_i)
            loss_val = loss.cost_sensitive_loss(model_output, y_i, b00, b01, b10, b11)
            loss_val += lam * model.regularize()
            total_loss += loss_val
            loss_val.backward()
            optimizer.step()
        loss_curve.append(total_loss)
    return model, loss_curve


