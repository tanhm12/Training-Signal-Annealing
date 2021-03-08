import torch
import time
import numpy as np

from torch import nn
from tqdm import tqdm
from TSA import TSA_CrossEntropyLoss, TSA

torch.random.manual_seed(seed=222)

device = 'cpu'
num_labels = 2
num_data = 3200
batch_size = 32
epochs = 5

def make_model():
    model = nn.Sequential(
        nn.Linear(10, 200),
        nn.Linear(200, num_labels)
    )
    return model


model = make_model()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

x = torch.rand((num_data, 10), device=device)
y = torch.randint(0, num_labels, (num_data,), dtype=torch.long, device=device)


def test_criterion(x, y, criterion, print_freq=0.05):
    best_dev_loss = 1e9
    model.train()

    print_after = int(print_freq * len(x))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, )
    for epoch in range(epochs):
        print_counter = 0
        total_loss = []
        print('epoch:', epoch)
        for i in tqdm(range(0, len(x), batch_size)):
            logits = model(x[i: i + batch_size])
            targets = y[i: i + batch_size]
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss.append(loss.item())
            if i > print_counter:
                print('step: {}, loss: {}, total loss: {}'.format(i, loss.item(), np.mean(total_loss)))
                print_counter += print_after
        scheduler.step()
        print('train loss:', np.mean(total_loss))


# test_criterion(x, y, nn.CrossEntropyLoss())
test_criterion(x, y, TSA_CrossEntropyLoss(TSA(T=num_data/batch_size * epochs, K=2)))




