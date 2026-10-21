import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 16

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
epochs = 200

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    average_loss = total_loss / len(train_dataset)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")


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
model.eval()

with torch.no_grad():
    train_predictions = (torch.sigmoid(model(X_train)) >= 0.5).float()
    test_predictions = (torch.sigmoid(model(X_test)) >= 0.5).float()

    train_accuracy = (train_predictions == y_train).float().mean()
    test_accuracy = (test_predictions == y_test).float().mean()

print(f"Training accuracy: {train_accuracy.item() * 100:.2f}%")
print(f"Test accuracy: {test_accuracy.item() * 100:.2f}%")