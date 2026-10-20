import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("TitanicModel/Titanic-Dataset.csv", usecols=["Age", "Sex", "Pclass", "Survived"])
df = df.dropna(subset=['Age', 'Pclass', 'Sex', 'Survived']) #specifing might be not necessary but just to be sure
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Age", "Sex", "Pclass"]].values
y = df[["Survived"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class TitanicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.network(x)
model = TitanicNN()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 2000

for epoch in range(epochs):

    model.train()

    predictions = model(X_train)

    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    logits = model(X_test)

    probabilities = torch.sigmoid(logits)

    predicted_classes = (probabilities >= 0.5).float()

    accuracy = (predicted_classes == y_test).float().mean()

    print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")

new_passenger = pd.DataFrame(
    [[25, 0, 3]],
    columns=["Age", "Sex", "Pclass"]
)

new_passenger = scaler.transform(new_passenger)
new_passenger = torch.tensor(new_passenger, dtype=torch.float32)

with torch.no_grad():
    probability = torch.sigmoid(model(new_passenger))

print(f"Survival probability: {probability.item():.2f}")
