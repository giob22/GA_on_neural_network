import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_network.nn_engine import neural_network
from neural_network.nn_layer import leaky_relu, softmax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, shift
import numpy as np
import pickle
import random

RUNS          = 15
EPOCHS        = 1000
LR_START      = 0.02   # learning rate iniziale
LR_DECAY      = 0.97   # moltiplicatore ogni 50 epoche
LR_MIN        = 0.003  # soglia minima
SAVE_PATH     = Path(__file__).resolve().parent / "test_app" / "model.pkl"


def augment_sample(img8x8, rng):
    """
    Trasforma un campione 8×8 (valori 0–16) simulando variazioni realistiche:
    - blur gaussiano (simula il pennello morbido del canvas)
    - jitter spaziale ±1 pixel (simula diverse centrature)
    - scaling dell'intensità ×0.7–1.3 (simula diversa pressione del pennello)
    - rumore gaussiano leggero
    """
    img = img8x8.reshape(8, 8).astype(float)

    # blur: simula il falloff del pennello
    sigma = rng.uniform(0.2, 0.8)
    img = gaussian_filter(img, sigma=sigma)

    # jitter spaziale: shift ±1 px con bordi a 0
    dr = rng.integers(-1, 2)
    dc = rng.integers(-1, 2)
    img = shift(img, [dr, dc], mode='constant', cval=0.0)

    # scaling intensità
    scale = rng.uniform(0.70, 1.30)
    img = img * scale

    # rumore gaussiano leggero
    img = img + rng.normal(0, 0.5, img.shape)

    return np.clip(img, 0, 16).reshape(-1)


def build_augmented_set(X_raw, Y_raw, rng, n_aug=4):
    """
    Costruisce il training set aumentato:
    - campioni originali
    - n_aug versioni augmentate per campione
    - versione binarizzata (simula vecchia UI binaria)
    """
    parts_X = [X_raw]
    parts_Y = [Y_raw]

    for _ in range(n_aug):
        X_aug = np.array([augment_sample(x, rng) for x in X_raw])
        parts_X.append(X_aug)
        parts_Y.append(Y_raw)

    # binarizzata: pixel >= 8 → 16, altrimenti 0
    X_bin = np.where(X_raw >= 8.0, 16.0, 0.0)
    parts_X.append(X_bin)
    parts_Y.append(Y_raw)

    return np.vstack(parts_X), np.vstack(parts_Y)


if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target
    Y_onehot = np.eye(10)[y]

    best_accuracy = 0.0

    for run in range(RUNS):
        seed = random.randint(0, 1000)
        rng = np.random.default_rng(seed)

        X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
            X, Y_onehot, test_size=0.2, random_state=seed, stratify=y
        )

        X_train_aug, Y_train_aug = build_augmented_set(X_train_raw, Y_train, rng)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled  = scaler.transform(X_test_raw)

        nn = neural_network(
            hidden_config=[(64, leaky_relu), (32, leaky_relu)],
            size_input=64,
            size_output=10,
            learning_rate=LR_START,
            output_function=softmax,
            rng=rng,
        )

        for epoch in range(EPOCHS):
            # learning rate decay ogni 50 epoche
            if epoch > 0 and epoch % 50 == 0:
                nn.lr = max(LR_MIN, nn.lr * LR_DECAY)

            perm = np.random.permutation(len(X_train_scaled))
            for idx in perm:
                nn.feedback(X_train_scaled[idx], Y_train_aug[idx])

        correct = sum(
            1 for j in range(len(X_test_scaled))
            if np.argmax(nn.feedforward(X_test_scaled[j])['guess']) == np.argmax(Y_test[j])
        )
        accuracy = correct / len(X_test_scaled)
        final_lr = nn.lr

        print(f"run {run+1:2d}/{RUNS}  seed={seed:4d}  acc={accuracy:.4f}  lr_final={final_lr:.5f}", end="")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            with open(SAVE_PATH, "wb") as f:
                pickle.dump({"nn": nn, "scaler": scaler}, f)
            print("  ← saved")
        else:
            print()

    print(f"\nbest accuracy: {best_accuracy:.4f}  →  {SAVE_PATH}")
