from sklearn.datasets import load_digits
import numpy as np
import pickle

with open("model.pkl", "rb") as f:
    data = pickle.load(f)

nn     = data["nn"]
scaler = data["scaler"]

digits = load_digits()
X, y   = digits.data, digits.target
# test su un singolo campione
idx     = 7
sample  = X[idx]
label   = y[idx]
print()


x_scaled = scaler.transform([sample])
print(sample)
print(x_scaled)
pred     = np.argmax(nn.feedforward(x_scaled[0])['guess'])

print(f"Campione #{idx} — reale: {label}, predetto: {pred}, {'OK' if pred == label else 'ERRORE'}")

# accuracy sull'intero dataset
correct = 0
for i in range(len(X)):
    x_scaled = scaler.transform([X[i]])
    pred = np.argmax(nn.feedforward(x_scaled[0])['guess'])
    if pred == y[i]:
        correct += 1

print(f"Accuracy su tutto il dataset: {correct / len(X) * 100:.2f}%")
