import random
from .nn_layer import *
from .nn_engine import neural_network

class GeneticAlgorithm:

    def __init__(self, population_size, generations, mutation_rate, tournament_size, epochs, learning_rate, n_feature, n_output, K, lambda_, X_Train, Y_Train, X_val, Y_val):
        """
        @brief Inizializza l'algoritmo genetico per la ricerca automatica dell'architettura.

        @param population_size (int) Numero di individui per generazione.
        @param generations (int) Numero di generazioni evolutive da eseguire.
        @param mutation_rate (float) Probabilità [0, 1] che ogni gene venga mutato.
        @param tournament_size (int) Numero di individui che partecipano a ogni torneo di selezione.
        @param epochs (int) Epoche di backpropagation per addestrare ogni rete durante la fitness.
        @param learning_rate (float) Learning rate della backpropagation interna.
        @param n_feature (int) Numero di feature in input (corrisponde a size_input della rete).
        @param n_output (int) Numero di classi in output (corrisponde a size_output della rete).
        @param K (int) Numero di valutazioni indipendenti per individuo; la fitness finale è la media.
        @param lambda_ (float) Coefficiente di penalità sulla complessità nella funzione di fitness.
        @param X_Train (numpy.ndarray) Feature del training set, forma (n_campioni, n_feature).
        @param Y_Train (numpy.ndarray) Label one-hot del training set, forma (n_campioni, n_output).
        @param X_val (numpy.ndarray) Feature del validation set.
        @param Y_val (numpy.ndarray) Label one-hot del validation set.
        @note I dati devono essere già normalizzati e splittati prima di essere passati.
        """
        self.n_feature = n_feature
        self.n_output = n_output
        self.population_size = population_size
        # numero di individui
        self.generations = generations
        # quante generazioni far evolvere
        self.mutation_rate = mutation_rate
        # mutation rate
        self.tournament_size = tournament_size
        # quanti individui partecipano ad ogni torneo di selezione
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.lambda_ = lambda_


        self.X_train = X_Train
        self.Y_train = Y_Train
        self.X_val = X_val
        self.Y_val = Y_val
        # Dati già slittati e normalizzati

        self.hidden_functions = [relu, leaky_relu, sigmoid]
        self.output_functions = [softmax, linear, sigmoid]

        # Serve per evitare che la fitness sia falsata di troppo dalla scelta casuale dei pesi
        self.K = K


    def _genera_individuo(self):
        """
        @brief Genera un cromosoma casuale che rappresenta un'architettura di rete neurale.

        Sceglie casualmente un numero di hidden layer tra 1 e 4; per ognuno
        sorteggia neuroni (4–32) e funzione di attivazione da hidden_functions.

        @return (list[tuple[int, callable]]) Lista di tuple (n_neuroni, funzione),
                una per ogni hidden layer.
        """
        cromosoma = []
        n_hidden_layer = random.randint(1, 4)
        for _ in range(0, n_hidden_layer):
            cromosoma.append((random.randint(4, 32), random.choice(self.hidden_functions)))

        return cromosoma


    def _complessita(self, individuo):
        """
        @brief Calcola il numero totale di parametri (pesi + bias) di un'architettura.

        Usato come penalità nella funzione di fitness per scoraggiare reti troppo grandi.

        @param individuo (list[tuple[int, callable]]) Cromosoma che descrive l'architettura.
        @return (int) Numero totale di parametri della rete corrispondente.
        """
        params = 0
        prev = self.n_feature # n of inputs
        for n_neuroni, _ in individuo:
            params += n_neuroni * prev + n_neuroni # pesi + bias
            prev = n_neuroni
        params += self.n_output * prev + self.n_output # output layer
        return params

    def _fitness(self, individuo):
        """
        @brief Valuta la qualità di un individuo tramite addestramento e misura dell'accuracy.

        Ripete K volte: costruisce la rete con quell'architettura (pesi casuali),
        la addestra sul training set per self.epochs epoche, valuta l'accuracy sul
        validation set. La fitness finale penalizza la complessità dell'architettura.

        @param individuo (list[tuple[int, callable]]) Cromosoma da valutare.
        @return (tuple[float, float]) Coppia (fitness, accuracy) dove:
                - fitness = accuracy - lambda_ * n_params
                - accuracy = media delle K valutazioni sul validation set.
        @note K valutazioni indipendenti riducono la varianza dovuta all'inizializzazione casuale.
        """
        sum_accuracy = 0
        for _ in range(0, self.K):
            n = neural_network(individuo, self.n_feature, self.n_output, self.learning_rate, softmax)
            # ADDESTRAMENTO → training set
            for _ in range(0, self.epochs):
                x = random.randint(0, len(self.Y_train) - 1)
                n.feedback(self.X_train[x], self.Y_train[x])
            # VALUTAZIONE → testing set
            correct = 0
            for i in range(0, len(self.Y_val)):
                guess = n.feedforward(self.X_val[i])['guess']
                if np.argmax(guess) == np.argmax(self.Y_val[i]):
                    correct += 1
            sum_accuracy += correct / len(self.X_val)

        accuracy = sum_accuracy / self.K

        penalita = self._complessita(individuo)
        fitness = accuracy - self.lambda_ * penalita

        return (fitness, accuracy)

    def _selezione(self, popolazione, fitness_scores):
        """
        @brief Seleziona un individuo tramite torneo.

        Estrae casualmente tournament_size individui dalla popolazione e
        restituisce quello con la fitness più alta.

        @param popolazione (list) Lista dei cromosomi della generazione corrente.
        @param fitness_scores (list[float]) Lista di fitness corrispondente a ogni individuo.
        @return (list[tuple[int, callable]]) Il cromosoma vincitore del torneo.
        """
        idx_t = random.sample(range(0, len(popolazione)), k=self.tournament_size)
        idx_sorted = sorted(idx_t, key=lambda x: fitness_scores[x], reverse=True)
        return popolazione[idx_sorted[0]]


    def _crossover(self, genitore1, genitore2):
        """
        @brief Genera un figlio combinando due genitori con punto di taglio variabile.

        Sceglie un punto di taglio indipendente per ciascun genitore e costruisce
        il figlio con i primi t1 layer di genitore1 e i layer da t2 in poi di genitore2.

        @param genitore1 (list[tuple[int, callable]]) Primo cromosoma genitore.
        @param genitore2 (list[tuple[int, callable]]) Secondo cromosoma genitore.
        @return (list[tuple[int, callable]]) Cromosoma figlio.
        @note Se il crossover produce un figlio vuoto, viene inserito un gene casuale
              dal pool dei due genitori.
        """
        t1 = random.randint(0, len(genitore1))
        t2 = random.randint(0, len(genitore2))

        # caso in cui t1 = 0 e t2 = len(genitore2)
        figlio = genitore1[:t1] + genitore2[t2:]
        if len(figlio) == 0:
            pool = genitore1 + genitore2
            figlio.append(random.choice(pool))
        return figlio

    def _mutazione(self, individuo):
        """
        @brief Applica mutazioni casuali a un cromosoma.

        Per ogni gene, con probabilità mutation_rate, applica una delle tre mutazioni
        con pesi [20, 20, 1]:
        - tipo 0: cambia il numero di neuroni del layer (4–32);
        - tipo 1: cambia la funzione di attivazione;
        - tipo 2: aggiunge o rimuove un layer intero.

        @param individuo (list[tuple[int, callable]]) Cromosoma da mutare.
        @return (list[tuple[int, callable]]) Cromosoma mutato (copia, l'originale non viene modificato).
        @note Dopo una rimozione di layer il loop si interrompe con break perché
              gli indici non sono più validi.
        @note Un layer viene rimosso solo se l'individuo ha più di un layer.
        """
        individuo = [layer_ for layer_ in individuo]
        for i in range(0, len(individuo)):
            if random.random() < self.mutation_rate:
                new_layer = ()
                mut = random.choices([0, 1, 2], [20, 20, 1])[0]
                if mut == 0: # mutano il numero di neuroni
                    new_layer = (random.randint(4, 32), individuo[i][1])
                    individuo[i] = new_layer
                elif mut == 1: # muta la funzione di attivazione
                    new_layer = (individuo[i][0], random.choice(self.hidden_functions))
                    individuo[i] = new_layer
                else:

                    if random.randint(0, 1) == 0 and len(individuo) > 1:
                        pos = random.randint(0, len(individuo) - 1)

                        del individuo[pos]
                        break # necessario perché dopo l'elimiazione gli indici non sono più validi
                    else:

                        pos = random.randint(0, len(individuo))
                        individuo.insert(pos, (random.randint(4, 32), random.choice(self.hidden_functions)))
        return individuo


    def run(self):
        """
        @brief Esegue il ciclo evolutivo principale e restituisce i risultati.

        Ad ogni generazione: valuta la fitness di tutti gli individui, aggiorna
        il miglior individuo assoluto, salva le statistiche, genera la nuova
        popolazione tramite elitismo + crossover + mutazione.

        @return (tuple) Tupla con sei elementi:
                - best_individuo (list[tuple[int, callable]]): architettura ottimale trovata.
                - best_fitness (float): fitness del miglior individuo.
                - best_accuracy (float): accuracy del miglior individuo sul validation set.
                - storia_best_fitness (list[float]): fitness migliore per ogni generazione.
                - storia_best_accuracy (list[float]): accuracy migliore per ogni generazione.
                - storia_mean_accuracy (list[float]): accuracy media della popolazione per generazione.
        @note Il miglior individuo di ogni generazione viene preservato nella generazione
              successiva (elitismo).
        """
        # Cosa deve restituire:
        # - miglior architettura trovata
        # - storia della fitness migliore generazione
        # - storia della fitness media per generazione
        popolazione = [self._genera_individuo() for _ in range(0, self.population_size)]

        storia_best_fitness = [] # migliore per generazione della fitness
        storia_best_accuracy = [] # migliore per generazione della accuracy
        storia_mean_accuracy = [] # media per generazione dell'accuracy

        best_individuo = None
        best_fitness = 0

        for i in range(0, self.generations):

            # valutiamo la fitness di ogni individuo della popolazione
            risultati = [self._fitness(individuo) for individuo in popolazione]
            fitness_scores  = [r[0] for r in risultati]
            accuracy_scores = [r[1] for r in risultati]


            # Aggiornamento del migliore in assoluto
            idx_best = np.argmax(fitness_scores)
            if fitness_scores[idx_best] > best_fitness:
                best_fitness = fitness_scores[idx_best]
                best_individuo = popolazione[idx_best]
                best_accuracy = accuracy_scores[idx_best]


            # salviamo le statistiche
            storia_best_fitness.append(max(fitness_scores))
            storia_best_accuracy.append(max(accuracy_scores))
            storia_mean_accuracy.append(np.mean(accuracy_scores))

            # stampa dell'avanzamento
            print(f"[gen: {i:>3}] best fitness: {round(storia_best_fitness[-1]*100, 2):>5}% | best accuracy: {round(storia_best_accuracy[-1] * 100, 2):>5}% | mean accuracy: {round(storia_mean_accuracy[-1] * 100, 2):>5}% ")



            # Nuova popolazione
            new_population = []

            if best_individuo is not None:
                new_population.append(best_individuo)

            while len(new_population) < self.population_size:
                # scelta dei genitori
                genitore1 = self._selezione(popolazione, fitness_scores)
                genitore2 = self._selezione(popolazione, fitness_scores)

                # generiamo il figlio
                figlio = self._crossover(genitore1, genitore2)

                # mutazioni
                figlio = self._mutazione(figlio)

                new_population.append(figlio)
            popolazione = new_population

        return (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy)
