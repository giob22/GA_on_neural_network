from neural_network import *
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)






import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_neural_network(ax, individuo, max_neurons_shown=32):
    """
    Disegna la rete neurale come grafo a strati su un asse matplotlib.
    Si adatta automaticamente alle dimensioni dell'asse senza distorcere i cerchi.

    Parametri:
    - ax:                 asse matplotlib su cui disegnare
    - individuo:          lista di tuple (n_neuroni, funzione) — cromosoma migliore
                          le funzioni devono essere ancora oggetti funzione, non stringhe
    - max_neurons_shown:  soglia oltre la quale si usano i puntini di sospensione
    """

    # ── Costruzione dei layer ────────────────────────────────────────────
    layers = [
        {'count': 4, 'label': 'Input', 'color': '#B5D4F4', 'edge': '#185FA5'},
    ]
    for i, (n, f) in enumerate(individuo):
        layers.append({
            'count': n,
            'label': f'Hidden {i+1}\n{f.__name__}',
            'color': '#9FE1CB',
            'edge':  '#0F6E56'
        })
    layers.append(
        {'count': 3, 'label': 'Output\nsoftmax', 'color': '#F5C4B3', 'edge': '#993C1D'}
    )

    n_layers  = len(layers)
    SLOT_H    = 1.0    # altezza di una slot verticale (unità plot)
    NEURON_R  = 0.28   # raggio cerchio in unità y
    DOT_R     = 0.06   # raggio puntino in unità y
    X_STEP    = 2.2    # distanza orizzontale tra layer (unità x)

    # ── Helpers geometrici ───────────────────────────────────────────────

    def slots_for_layer(count):
        if count <= max_neurons_shown:
            return count
        return max_neurons_shown + 2  # neuroni visibili + slot puntini

    def neuron_positions(count, layer_slots):
        """
        Restituisce lista di (y, tipo) con tipo 'neuron' o 'dots'.
        Le posizioni sono centrate attorno a y=0.
        """
        total_h = layer_slots * SLOT_H
        start_y = -total_h / 2 + SLOT_H / 2

        if count <= max_neurons_shown:
            return [(start_y + i * SLOT_H, 'neuron') for i in range(count)]

        half = max_neurons_shown // 2
        positions = []
        for i in range(half):
            positions.append((start_y + i * SLOT_H, 'neuron'))
        dot_y = start_y + half * SLOT_H
        positions.append((dot_y, 'dots'))
        for i in range(half, max_neurons_shown):
            positions.append((start_y + (i + 1) * SLOT_H, 'neuron'))
        return positions

    # ── Calcolo dimensioni canvas ────────────────────────────────────────
    max_slots = max(slots_for_layer(l['count']) for l in layers)
    total_h   = max_slots * SLOT_H
    x_min     = -0.5
    x_max     = (n_layers - 1) * X_STEP + 0.5
    y_min     = -total_h / 2 - 0.8
    y_max     =  total_h / 2 + 0.8

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    rx_neuron = NEURON_R
    rx_dot    = DOT_R

    # ── Precomputa posizioni per tutti i layer ───────────────────────────
    all_positions = []
    for layer in layers:
        slots     = slots_for_layer(layer['count'])
        positions = neuron_positions(layer['count'], slots)
        all_positions.append(positions)

    # ── Disegna connessioni ──────────────────────────────────────────────
    for li in range(n_layers - 1):
        x1    = li * X_STEP
        x2    = (li + 1) * X_STEP
        pos_a = [y for y, t in all_positions[li]   if t == 'neuron']
        pos_b = [y for y, t in all_positions[li+1] if t == 'neuron']
        for ya in pos_a:
            for yb in pos_b:
                ax.plot(
                    [x1 + rx_neuron, x2 - rx_neuron],
                    [ya, yb],
                    color='#B4B2A9', linewidth=0.4, zorder=1, alpha=0.55
                )

    # ── Etichette interne per input e output ─────────────────────────────
    neuron_labels = {
        0:           ['x₁', 'x₂', 'x₃', 'x₄'],
        n_layers - 1: ['y₁', 'y₂', 'y₃']
    }

    # ── Disegna neuroni e puntini ────────────────────────────────────────
    for li, (layer, positions) in enumerate(zip(layers, all_positions)):
        x          = li * X_STEP
        neuron_idx = 0

        for y, tipo in positions:
            if tipo == 'neuron':
                ellipse = mpatches.Ellipse(
                    (x, y),
                    width=rx_neuron * 2,
                    height=NEURON_R * 2,
                    facecolor=layer['color'],
                    edgecolor=layer['edge'],
                    linewidth=1.2,
                    zorder=3
                )
                ax.add_patch(ellipse)

                # testo dentro il cerchio solo per input e output
                if li in neuron_labels and neuron_idx < len(neuron_labels[li]):
                    ax.text(
                        x, y,
                        neuron_labels[li][neuron_idx],
                        ha='center', va='center',
                        fontsize=6.5, fontweight='bold',
                        color=layer['edge'], zorder=4
                    )
                neuron_idx += 1

            else:  # puntini di sospensione
                for dy in [-0.18, 0, 0.18]:
                    dot = mpatches.Ellipse(
                        (x, y + dy),
                        width=rx_dot * 2,
                        height=DOT_R * 2,
                        facecolor='#888780',
                        zorder=3
                    )
                    ax.add_patch(dot)

        # ── Label sotto ogni layer ───────────────────────────────────────
        ax.text(
            x, y_min + 0.05,
            layer['label'],
            ha='center', va='bottom',
            fontsize=7.5, color='#444441',
            linespacing=1.4
        )

        # ── Conteggio neuroni sopra ogni layer ───────────────────────────
        ax.text(
            x, y_max - 0.05,
            str(layer['count']),
            ha='center', va='top',
            fontsize=7.5, color='#5F5E5A'
        )

    ax.set_title(
        'Architettura ottimale trovata',
        fontsize=9, pad=6, color='#2C2C2A'
    )











def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

def number_params(input_size, individuo, output_size):
    params = 0
    prev = input_size
    for neuroni, _ in individuo:
        params += neuroni * prev + neuroni
        prev = neuroni
    params += output_size * prev + output_size
    return params



K = 5 # numero di addestramenti per individuo

POPULAZION_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5
EPOCHS =  300 #150
LEARNING_RATE = 0.01
LAMBDA_ = 0.0005

EPOCHS_BASELINE = K * EPOCHS


if __name__ == "__main__":

    # load del dataset
    dataset = load_iris()
    
    x = dataset.data
    y = dataset.target

    # numero di feature
    input_size = x.shape[1]

    # numero di classi
    n_classi = len(np.unique(y))


    output_function = softmax

    
    nome_riga = [r for r in dataset.DESCR.split('\n') if r.strip() and not r.startswith('..')]
    DATASET_NAME = nome_riga[0].strip()   # → "Iris plants dataset"
    


    
    
    

    # split train/validation

    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)


    # usiamo random_state=42 per fissare il seed in modo che il risultato sia riproducibile
    # ogni volta che eseguo il programma ottengo lo stesso split, mi permette di confrontare i risultati tra run diverse

    # Normalizzazione
    # X ← (X - media(X)) / std(X)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    
    y_train = one_hot(y_train, n_classes=n_classi)
    y_val = one_hot(y_val, n_classes=n_classi)


    # BASELINE con backpropagation classica
    # rete neurale addestrata per lo stesso quantitativo di epoche 
    # mi permette di confrontare il risultato del GA
    cromosoma_baseline = [(8,leaky_relu),(8,leaky_relu)]
    rete_baseline = neural_network(hidden_config=cromosoma_baseline,size_input=input_size , size_output=n_classi, learning_rate=0.01, output_function=output_function)

    for i in range(0, EPOCHS_BASELINE):
        idx = random.randint(0, len(x_train) - 1)
        rete_baseline.feedback(x_train[idx], y_train[idx])

    # valutiamo la baseline sul validation set
    correct = 0
    for i in range(0, len(x_val)):
        guess = rete_baseline.feedforward(x_val[i])['guess']
        if np.argmax(guess) == np.argmax(y_val[i]):
            correct += 1
    accuracy_baseline = correct/len(x_val)
    logger.info(f"accuracy della rete baseline: {round((accuracy_baseline) * 100, 2)}%\n#params: {number_params(input_size, cromosoma_baseline, n_classi)}")

    # ESECUZIONE DEL Genetic Algorithm

    ga = GeneticAlgorithm(population_size=POPULAZION_SIZE,
                          generations=GENERATIONS,
                          mutation_rate=MUTATION_RATE,
                          tournament_size=TOURNAMENT_SIZE,
                          epochs=EPOCHS,
                          learning_rate=LEARNING_RATE,
                          lambda_=LAMBDA_,
                          n_feature=input_size,
                          n_output= n_classi,
                          K=K,
                          X_Train=x_train, Y_Train=y_train,
                          X_val=x_val, Y_val=y_val)
    
    (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy) = ga.run()
   
   
    fig, axes = plt.subplot_mosaic(
    [['fitness',      'accuracy'],
     ['fitness',     'architettura'],
     ['rete',         'rete']],
    figsize=(10, 11),
    gridspec_kw={'height_ratios': [1, 1, 2],'width_ratios': [1, 1]}
    )


    ax_fit  = axes['fitness']
    ax_acc  = axes['accuracy']
    ax_arch = axes['architettura']
    ax_rete = axes['rete']

    plt.tight_layout()
    draw_neural_network(axes['rete'], best_individuo)


    best_individuo = [(neuroni, funzione.__name__) for neuroni, funzione in best_individuo]
    logger.info(f"Miglior architettura trovata:\n{best_individuo}")
    logger.info(f"Fitness:{round(best_fitness * 100,2)}")
    logger.info(f"Accuracy:{round(best_accuracy * 100,2)}%")
    logger.info(f"Numero di layer={len(best_individuo)}")
    logger.info(f"#parametri={number_params(input_size,best_individuo, n_classi)}")

    # trovo nella storia in che generazione si posizione il best_individuo
    idx_best = np.argmax(storia_best_fitness)
    


    
    linespace_gen = list(range(1, GENERATIONS + 1))

    best_fitness_pct  = [v * 100 for v in storia_best_fitness]
    best_accuracy_pct = [v * 100 for v in storia_best_accuracy]
    mean_accuracy_pct = [v * 100 for v in storia_mean_accuracy]
    baseline_pct      = accuracy_baseline * 100


    # fig, (ax_acc, ax_fit) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Algoritmo Genetico — ricerca dell'architettura ottimale\n"
        f"({DATASET_NAME}, classificazione {n_classi} classi)",
        fontsize=13, fontweight='bold'
    )

    # --- Grafico superiore: Accuracy ---
    ax_acc.plot(linespace_gen, best_accuracy_pct,
                color="steelblue", linewidth=2,
                label="Miglior accuracy della generazione")
    ax_acc.plot(linespace_gen, mean_accuracy_pct,
                color="seagreen", linewidth=1.5, linestyle='--',
                label="Accuracy media della popolazione")
    ax_acc.axhline(baseline_pct, color="crimson", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop ({baseline_pct:.1f}%)")

    # annotazione valore finale best accuracy
    ax_acc.annotate(
        f"{best_accuracy_pct[idx_best]:.1f}%",
        xy=(idx_best + 1, best_accuracy_pct[idx_best]),
        xytext=(0, 12), textcoords='offset points',
        fontsize=9, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2)
    )

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(0, 110)
    ax_acc.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        "Accuratezza — quanto bene classifica la rete migliore e la popolazione media",
        fontsize=10
    )

    # --- Grafico inferiore: Fitness (accuracy penalizzata dalla complessità) ---
    ax_fit.plot(linespace_gen, best_fitness_pct,
                color="darkorange", linewidth=2,
                label="Miglior fitness della generazione")
    ax_fit.axhline(baseline_pct, color="crimson", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop ({baseline_pct:.1f}%)")

    ax_fit.annotate(
        f"{best_fitness_pct[idx_best]:.1f}",
        xy=(idx_best + 1, best_fitness_pct[idx_best]),
        xytext=(0, 15), textcoords='offset points',
        fontsize=9, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2)
    )

    ax_fit.set_xlabel("Generazione", fontsize=11)
    ax_fit.set_ylabel("Fitness", fontsize=11)
    ax_fit.set_xlim(1, GENERATIONS)
    ax_fit.set_ylim(0, 110)
    ax_fit.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_title(
        f"Fitness = accuracy − λ·complessità  (λ={LAMBDA_}) — premia reti accurate e semplici",
        fontsize=10
    )

    # grafico a destra: architettura
    ax_arch.axis('off')
    testo = "Input: 4 neuroni\n\n"
    for i, (neuroni, funzione) in enumerate(best_individuo):
            testo += f"Hidden layer {i + 1}: {neuroni} neuroni → {funzione}\n"
    testo += f"\n Output: 3 neuroni → {output_function.__name__}"
    testo += f"\n\nAccuracy: {round(best_accuracy * 100,2)}%"
    testo += f"\nFitness: {round(best_fitness * 100, 2)}"
    testo += f"\n#parametri: {number_params(input_size, best_individuo, n_classi)}"
    ax_arch.text(0.5,0.5, testo, transform=ax_arch.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax_arch.set_title("Migliore architettura trovata")



    plt.tight_layout()
    plt.savefig('img/risultati.png', dpi=150)
    plt.show()
    


