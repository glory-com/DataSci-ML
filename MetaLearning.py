import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_features, in_features) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_features))
        self.w2 = nn.Parameter(torch.randn(out_features, hidden_features) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(out_features))

    def functional_forward(self, x, weights):
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        return x

    def forward(self, x):
        params = [self.w1, self.b1, self.w2, self.b2]
        return self.functional_forward(x, params)

class MAML:
    def __init__(self):
        self.model = SimpleNet(in_features=X.shape[1], out_features=2).to(device)
        self.inner_lr = 0.01
        self.outer_lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.inner_steps = 5
        self.outer_steps = 10

    def inner_loop(self, x, y):
        fast_weights = list(self.model.parameters())
        for _ in range(self.inner_steps):
            preds = self.model.functional_forward(x, fast_weights)
            loss = self.loss_fn(preds, y)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        return fast_weights

    def outer_loop(self, dataloader):
        for i in range(self.outer_steps):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                fast_weights = self.inner_loop(x, y)
                preds = self.model.functional_forward(x, fast_weights)
                loss = self.loss_fn(preds, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"-----第{i+1}轮 完成-----")

    def predict(self, dataloader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(device)
                preds = self.model(x)
                all_preds.append(preds.argmax(dim=1).cpu())
        return torch.cat(all_preds, dim=0)

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

def clean(df):
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
            if col != "Personality":
                df[col] = df[col].map({"Yes": 1, "No": 0})
            else:
                df[col] = df[col].map({"Extrovert": 1, "Introvert": 0})
        else:
            df[col] = df[col].fillna(df[col].median())

clean(df_train)
clean(df_test)

idx = df_test["id"]
df_train.drop(columns=["id"], inplace=True)
df_test.drop(columns=["id"], inplace=True)

y = df_train["Personality"]
X = df_train.drop(columns=["Personality"])
X_test = df_test


X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
test_loader = DataLoader(X_test_tensor, batch_size=32)


maml = MAML()
maml.outer_loop(dataloader)
results = maml.predict(test_loader)

submission = pd.DataFrame({
    "id": idx,
    "Personality": results
})

submission.to_csv("submission.csv" , index = False)
print("Done!")