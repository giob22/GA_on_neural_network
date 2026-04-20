import numpy as np
import logging

# ── He / Xavier snippet (da sessione precedente) ─────────────────────────────
rng = np.random.default_rng(42)

def init_weights(row, col, mode):
    if mode == "he":
        std = np.sqrt(2 / col)
    elif mode == "xavier":
        std = np.sqrt(1 / col)
    return rng.normal(0, std, size=(row, col))

col = 8
row = 4

he      = init_weights(row, col, "he")
xavier  = init_weights(row, col, "xavier")
uniform = rng.uniform(-0.5, 0.5, size=(row, col))

for name, W in [("He     ", he), ("Xavier ", xavier), ("Uniform", uniform)]:
    print(f"{name} | std attesa: {np.sqrt(2/col) if 'He' in name else (np.sqrt(1/col) if 'Xav' in name else 0.289):.4f}"
          f" | std reale: {W.std():.4f}"
          f" | min: {W.min():.4f} max: {W.max():.4f}")


# ── LOGGING SNIPPET ───────────────────────────────────────────────────────────
#
# logging è il modulo standard Python per messaggi strutturati.
# Sostituisce print() con un sistema che:
#   - filtra messaggi per gravità (livello)
#   - aggiunge timestamp, nome modulo, ecc.
#   - può scrivere su file, console, o entrambi — senza cambiare il codice
#
# GERARCHIA DEI LIVELLI (dal meno al più grave):
#   DEBUG    → dettagli interni, disabilitati in produzione
#   INFO     → avanzamento normale (sostituisce quasi tutte le print)
#   WARNING  → qualcosa di inatteso ma non fatale
#   ERROR    → errore che blocca un'operazione
#   CRITICAL → errore che blocca tutto
#
# Il logger stampa solo i messaggi con livello >= quello impostato in basicConfig.
# Esempio: level=INFO → stampa INFO/WARNING/ERROR/CRITICAL, ignora DEBUG.

logging.basicConfig(
    level=logging.DEBUG,                            # livello minimo da mostrare
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    #        timestamp   livello          nome logger  messaggio
)

# Ogni modulo ha il suo logger isolato, identificato da __name__.
# In uno script standalone __name__ == "__main__".
# In un modulo importato __name__ == "neural_network.genetic_algorithm", ecc.
# Questo permette di silenziare selettivamente un singolo modulo.
logger = logging.getLogger(__name__)

logger.debug("Questo appare solo se level=DEBUG")
logger.info("Avanzamento normale — sostituisce print()")
logger.warning("Qualcosa di strano, ma si continua")
logger.error("Operazione fallita")
logger.critical("Sistema compromesso")

# ── SILENZIARE SELETTIVAMENTE UN MODULO ──────────────────────────────────────
#
# I logger formano una gerarchia basata sul nome (separato da "."):
#   root
#   └── neural_network
#       └── neural_network.genetic_algorithm
#
# getLogger("nome") restituisce sempre lo stesso oggetto per quel nome.
# Puoi prenderlo da qualsiasi punto del codice e cambiarne il livello.

# Silenzia completamente genetic_algorithm (WARNING e sopra passano ancora):
logging.getLogger("neural_network.genetic_algorithm").setLevel(logging.WARNING)

# Silenzia tutto il package neural_network:
logging.getLogger("neural_network").setLevel(logging.WARNING)

# Riattiva solo genetic_algorithm anche se il parent è silenziato:
logging.getLogger("neural_network.genetic_algorithm").setLevel(logging.DEBUG)

# Silenzia tutto tranne gli errori:
logging.getLogger("neural_network").setLevel(logging.ERROR)

# Ripristina comportamento di default (eredita dal root logger):
logging.getLogger("neural_network").setLevel(logging.NOTSET)

# ── CONFRONTO DIRETTO ─────────────────────────────────────────────────────────

# print:   nessun contesto, nessun filtro, sempre visibile
print("messaggio con print")

# logging: timestamp + livello + nome modulo inclusi automaticamente
logger.info("messaggio con logging")

# ── SCRIVERE SU FILE (senza toccare il codice applicativo) ────────────────────

file_handler = logging.FileHandler("prove_log.txt")
file_handler.setLevel(logging.WARNING)              # solo WARNING e sopra su file
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(file_handler)

logger.info("questo va solo a console")             # INFO: non passa il filtro del file
logger.warning("questo va su console E su file")   # WARNING: passa entrambi i filtri
