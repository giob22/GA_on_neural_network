import numpy as np
import math as m
import matplotlib.pyplot as plt
from nn import neural_network
import os

import sympy as sp

t = sp.symbols("t")

f_target = (sp.sin(t*2*sp.pi) * sp.cos(t*2*sp.pi*2.15) + 1)/2

# creo il dataset della sinusoide

X_train = np.linspace(0,1,100).tolist();
# faccio il tolist() altrimenti sarebbe un numpy.ndarray
Y_train = []

func_corretta = False
while not func_corretta:
    try:
        activation_func = input("Inserisci la activation function che vorresti utilizzare:\n\t- sigmoid\n\t- tanh\n\t- relu\n\t- leaky_r\n: ")
        if activation_func == "sigmoid" or activation_func == "tanh" or activation_func == "relu" or activation_func == "leaky_r":
            func_corretta = True
        else:
            raise ValueError("valore non valido, riprova!!")
    except ValueError as e:
        print("[ERROR]", e)
    except:
        print("[ERRORE GRAVE]")
        os._exit(1)

f_target_np = sp.lambdify(t, f_target)

for x in X_train:
    # moltiplico x per 2*pi per avere un periodo completo del sin
    valore = f_target_np(x)
    # valore = (m.cos(x*2*m.pi) + 1)/2
    # valore = 
    # valore = (m.cos(x*2*m.pi ) * m.sin(x*2*m.pi * 2) + 1)/2


    # caso: treno di gradini
    # if x<0.2:
    #     valore = 0
    # elif x < 0.4:
    #     valore = 0.2
    # elif x < 0.6:
    #     valore = 0.4
    # elif x < 0.8:
    #     valore = 0.6
    # else:
    #     valore = 0.8


    #sommo 1 e divido per 2 per avere tutti valori compresi tra 0 e 1
    Y_train.append(valore)

# preparo la rete e il grafico

'''
input (x) = 1
output (y) = 1
hidden_size = 32

1 layer intermedio
1 layer di input
1 layer di output
'''
p = neural_network(n_layer=3, n_input=1, n_output=1, lr=0.01, hidden_size=64, activation_function=activation_func)

plt.ion() 
# per permettere al codice sottostante di continuare ad eseguire dopo aver fatto il plot
# necessario per l'animazione

fig = plt.figure(figsize=(10,6))

# Utilizziamo gridspec per creare una griglia 2x2
gs = fig.add_gridspec(2,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])

ax3.set_title("Loss values per frame (MSE)")
testo_loss = ax3.text(.5, -0.15, '', transform=ax3.transAxes, 
                      ha='right', va='top', fontsize=12, color='blue',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


# fig, (ax1, ax2) = plt.subplots(2,2,figsize=(10,6))
# fig: rappresenta l'intera finestra o contenitore globale
# ax1 e ax2: è il sigolo grafico all'interno della figure
# ax1: per il grafico della predizione
# ax2: per visualizzare la matrice dei pesi


linea_predizione, = ax1.plot([],[], color='red', linewidth=3, label='Predizione della rete')
loss_values, = ax3.plot([],[], color='blue', linewidth=3, label='loss (performance del modello)')

im = ax2.imshow(p.weights[1], cmap='coolwarm', aspect='auto')
cbar = fig.colorbar(im,ax=ax2)

ax1.plot(X_train, Y_train, color='blue',alpha=0.5, label='Sinusoide Reale (Target)')

ax1.set_xlim(0,1)
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, linestyle='-',alpha=0.6)
ax1.legend()

# animazione dell'addestramento

epoche_per_frame = 2000
frame_totali = 1000

# creiamo i punti per una predizione fluida
x_plot = np.linspace(0,1,1000)
# inizializzo il vettore che conterrà i loss value
loss_plot = []

for frame in range(frame_totali):

    errore_cumulato = 0

    for _ in range(epoche_per_frame):
        # scegliamo un punto a caso delle x e addestriamo la rete
        x_casuale = np.random.randint(0,len(X_train))
        y_hat = p.train(X_train[x_casuale], Y_train[x_casuale])[0][0]

        # aggiungiamo l'errore quadratico della singola predizione 
        errore_cumulato += (Y_train[x_casuale] - y_hat)**2
    
    # calcolo dell'errore quadratico medio (MSE) di questo frame
    loss_medio_frame = errore_cumulato / epoche_per_frame

    loss_plot.append(loss_medio_frame)

    # aggiorno il grafico ax3
    loss_epochs = np.arange(len(loss_plot))

    loss_values.set_data(loss_epochs, loss_plot) 
    testo_loss.set_text(f"Loss medio attuale: {loss_medio_frame:.6f}")

    # Aggiorniamo dinamicamente gli assi del grafico loss altrimenti esce dalla vista
    ax3.set_xlim(0, max(10, len(loss_plot))) # Mostra almeno 10 epoche sull'asse x
    ax3.set_ylim(0, max(loss_plot) * 1.1 if max(loss_plot) > 0 else 1)     
    
    # calcolo la curva predetta dalla rete
    y_plot = []
    for val_x in x_plot:
        output, _ = p.prediction(np.array([val_x]).reshape(-1,1))
        y_plot.append(output[0][0])
    
    # aggiornamento della linea sul grafico
    linea_predizione.set_data(x_plot, y_plot)
    ax1.set_title(f"Predizione della rete\n[{f_target}]\nFrame: {frame:<10}Epoca: {frame * epoche_per_frame:<10}")

    # aggiorniamo il grafico della matrice dei pesi
    new_weights = p.weights[1]
    im.set_data(new_weights)
    im.set_clim(vmin=new_weights.min(), vmax=new_weights.max())
    ax2.set_title("Matrice dei pesi (Layer 1)")

    fig.canvas.draw() # ridisegna la tela
    fig.canvas.flush_events() # forza l'interfaccia a elaborare tutti gli eventi in sospeso
    plt.pause(0.01) # mette in pausa l'esecuzione dello script python → stiamo dando tempo al motore grafico di renderizzare e mostrare il risultato


plt.ioff()
print("Addestramento completato")
plt.show()