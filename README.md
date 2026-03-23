# Neural Network Python

Questo progetto è un'implementazione didattica di una rete neurale (Multi-Layer Perceptron) scritta completamente da zero. L'obiettivo è mostrare in modo pratico come funziona il processo di apprendimento "sotto il cofano", usando solo logica matriciale nuda e cruda con `numpy`, senza appoggiarsi a framework pronti come TensorFlow o PyTorch.

## Struttura del codice

Il progetto è diviso in due file principali:

### 1. `nn.py` - Il motore della rete
Contiene la classe `neural_network`, che fa tutto il lavoro sporco. Gestisce l'inizializzazione dei pesi, la funzione di attivazione (sigmoide), i passaggi di feedforward e la backpropagation per l'addestramento.

### 2. `sin.py` - L'addestramento visivo
È il main script da cui far partire tutto. Crea ed allena una rete ad imparare la forma di una sinusoide. Include un'animazione in tempo reale che mostra letteralmente la curva della rete adattarsi ai dati.

*(Nota: nel file troverai anche un pezzo commentato per far fittare alla rete un "treno di gradini" anziché un'onda, ottimo per testare le discontinuità).*

## Requisiti ed esecuzione

Moduli necessari:
- `numpy`
- `matplotlib`

Per avviarlo:
```bash
python sin.py
```

## Consigli per smanettarci su
- Prova a ridurre drasticamente il numero di nodi nascosti in `sin.py` (da 1000 a 10 o 20) e guarda come la rete fatica ad approssimare la curva.
- Modifica il `learning_rate` (`lr`): se è troppo grande la curva impazzirà, se è troppo piccolo ci metterà una vita a convergere.
- Prova a decommentare il treno di gradini e nota come la rete gestisca gli spigoli.
