import csv
import matplotlib.pyplot as plt
from neural_network import *
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── preprocessing (identico a main.py) ──────────────────────────────
def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

dataset = load_iris()
x = dataset.data
y = dataset.target
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
y_train = one_hot(y_train)
y_val   = one_hot(y_val)

# ── baseline ─────────────────────────────────────────────────────────
EPOCHS_BASELINE = 450
rete_baseline = neural_network([(8, relu), (8, relu)], 4, 3, 0.01, softmax)
for _ in range(EPOCHS_BASELINE):
    idx = random.randint(0, len(x_train) - 1)
    rete_baseline.feedback(x_train[idx], y_train[idx])
correct = sum(
    1 for i in range(len(x_val))
    if np.argmax(rete_baseline.feedforward(x_val[i])['guess']) == np.argmax(y_val[i])
)
accuracy_baseline = correct / len(x_val)
print(f"Baseline accuracy: {round(accuracy_baseline * 100, 2)}%")

# ── parametri fissi (identici al test lambda lineare per confronto diretto) ──
POPULATION_SIZE  = 20
GENERATIONS      = 30
LEARNING_RATE    = 0.01
EPOCHS           = 150
TOURNAMENT_SIZE  = 3
K                = 3
N_FEATURE        = 4   # feature Iris
N_OUTPUT         = 3   # classi Iris

# ── valori da testare ─────────────────────────────────────────────────
# Calibrati per penalità log10: log10(n_params su Iris) ≈ 1.8–3.7
# λ=0.05 → penalità ~0.09–0.19  (leggera)
# λ=0.15 → penalità ~0.27–0.56  (moderata)
# λ=0.30 → penalità ~0.54–1.11  (aggressiva)
lambdas = [0.0, 0.05, 0.15, 0.30]

# ── strutture per raccogliere i risultati ─────────────────────────────
csv_rows = []
all_storie = []

for lam in lambdas:
    print(f"\n{'='*50}")
    print(f"TEST lambda_log = {lam}")
    print(f"{'='*50}")

    ga = GeneticAlgorithm(
        population_size = POPULATION_SIZE,
        generations     = GENERATIONS,
        mutation_rate   = 0.2,
        tournament_size = TOURNAMENT_SIZE,
        epochs          = EPOCHS,
        learning_rate   = LEARNING_RATE,
        n_feature       = N_FEATURE,
        n_output        = N_OUTPUT,
        K               = K,
        lambda_         = lam,
        X_Train=x_train, Y_Train=y_train,
        X_val=x_val,     Y_val=y_val
    )

    (best_ind, best_fit, best_acc,
     storia_bf, storia_ba, storia_ma) = ga.run()

    n_layer  = len(best_ind)
    n_params = ga._complessita(best_ind)
    arch     = [(n, f.__name__) for n, f in best_ind]

    csv_rows.append({
        'lambda':            lam,
        'best_accuracy':     round(best_acc * 100, 2),
        'best_fitness':      round(best_fit * 100, 2),
        'n_layer':           n_layer,
        'n_params':          n_params,
        'accuracy_baseline': round(accuracy_baseline * 100, 2),
        'architettura':      str(arch)
    })

    all_storie.append({
        'label':     f"lambda={lam}",
        'storia_bf': storia_bf,
        'storia_ba': storia_ba,
        'storia_ma': storia_ma
    })

    print(f"Best accuracy:  {round(best_acc * 100, 2)}%")
    print(f"Best fitness:   {round(best_fit * 100, 2)}%")
    print(f"Architettura:   {arch}")
    print(f"n_params:       {n_params}  (log10={round(np.log10(n_params), 2)})")

# ── salva CSV ─────────────────────────────────────────────────────────
with open('tests/test_lambda_log.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print("\nCSV salvato: tests/test_lambda_log.csv")

# ── grafici comparativi ───────────────────────────────────────────────
gen = range(1, GENERATIONS + 1)

fig, axes = plt.subplot_mosaic(
    [['best_acc', 'mean_acc'],
     ['best_fit', 'mean_acc']],
    figsize=(14, 8)
)

for storia in all_storie:
    axes['best_acc'].plot(gen, [v * 100 for v in storia['storia_ba']], label=storia['label'])
    axes['best_fit'].plot(gen, [v * 100 for v in storia['storia_bf']], label=storia['label'])
    axes['mean_acc'].plot(gen, [v * 100 for v in storia['storia_ma']], label=storia['label'])

for ax_key in ['best_acc', 'mean_acc']:
    axes[ax_key].axhline(y=accuracy_baseline * 100, color='red',
                         linestyle='dashed', label='baseline')

axes['best_acc'].set_title("Best accuracy per generazione")
axes['best_acc'].set_ylabel("Accuracy (%)")
axes['best_acc'].set_xlabel("Generazione")
axes['best_acc'].legend()

axes['best_fit'].set_title("Best fitness per generazione  [fitness = acc − λ·log₁₀(params)]")
axes['best_fit'].set_ylabel("Fitness (%)")
axes['best_fit'].set_xlabel("Generazione")
axes['best_fit'].legend()

axes['mean_acc'].set_title("Mean accuracy per generazione")
axes['mean_acc'].set_ylabel("Accuracy (%)")
axes['mean_acc'].set_xlabel("Generazione")
axes['mean_acc'].legend()

plt.suptitle(
    f"Test λ con penalità log₁₀ — baseline: {round(accuracy_baseline*100, 2)}%",
    fontsize=14
)
plt.tight_layout()
plt.savefig('docs/studio/img/test_lambda_log.png', dpi=150)
plt.show()
print("Grafico salvato: docs/studio/img/test_lambda_log.png")
