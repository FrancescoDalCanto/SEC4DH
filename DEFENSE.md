# Difesa: Feature-Space Poison Detection

## Contesto: l'attacco che la difesa contrasta

Il progetto implementa un **clean-label feature-collision attack** (Shafahi et al., *"Poison Frogs!"*).

L'attaccante vuole che un'immagine maligna (`x_target`) venga classificata come **Normal** dal modello vittima. Per farlo:

1. Prende N immagini benigne con etichetta `Normal` (le *base images*).
2. Le perturba con un ottimizzatore (Adam) in modo che le loro rappresentazioni nello spazio delle feature di ResNet18 si avvicinino il più possibile a quelle di `x_target`.
3. La perturbazione è vincolata a un ball L-∞ di raggio `ε = 4/255`, quindi le immagini risultanti sono visivamente indistinguibili dagli originali.
4. Queste immagini avvelenate vengono inserite nel training set con l'etichetta corretta (`Normal = 1`).

Quando il modello vittima viene addestrato su questo dataset, il confine decisionale si sposta finché `x_target` (un'immagine **maligna**) viene erroneamente classificata come Normal.

---

## Principio della difesa

La difesa sfrutta **la stessa proprietà che l'attaccante ha ingegnerizzato**:

> I sample avvelenati, per costruzione, hanno feature molto vicine a quelle del target in feature space. I sample legittimi, invece, si trovano lontani dal target in quello stesso spazio.

Misurando la **cosine similarity** tra ogni sample sospetto e il target nel feature space di ResNet18, è possibile identificare e rimuovere i veleni prima del training.

---

## Come funziona passo per passo

### Step 1 — Calcolo del vettore del target

```python
target_feat = feature_extractor(x_target.to(device))
target_feat = F.normalize(target_feat, dim=1)  # vettore unitario
```

Il feature extractor (ResNet18 frozen, stesso usato dall'attaccante) trasforma `x_target` in un vettore a 512 dimensioni, che viene normalizzato a norma 1. Questo sarà il **punto di riferimento fisso**.

### Step 2 — Ispezione dei sample sospetti

Solo i sample con etichetta `suspect_label = 1` (Normal) vengono ispezionati. Gli altri vengono mantenuti direttamente, poiché l'attaccante inietta i veleni solo sotto questa etichetta.

```python
if int(label.item()) == suspect_label:
    feat = feature_extractor(img.unsqueeze(0).to(device))
    feat = F.normalize(feat, dim=1)
    sim = (feat * target_feat).sum().item()  # cosine similarity
```

Il prodotto scalare tra due vettori unitari equivale alla loro cosine similarity: vale `1` se sono identici, `0` se ortogonali.

### Step 3 — Soglia e rimozione

```python
if sim >= similarity_threshold:  # default: 0.90
    flagged += 1
    continue  # scarta il sample
```

Un sample viene marcato come veleno e rimosso se la sua cosine similarity col target supera la soglia (`0.90` di default). Il dataset restituito contiene solo i sample ritenuti puliti.

---

## Parametri chiave

| Parametro | Valore default | Significato |
|---|---|---|
| `suspect_label` | `1` (Normal) | Etichetta sotto cui vengono iniettati i veleni |
| `similarity_threshold` | `0.90` | Soglia oltre cui un sample è considerato veleno |

> **Nota sulla soglia:** Con `ε` piccolo (es. `4/255`) i veleni potrebbero non superare `0.90`. In quel caso abbassare la soglia a `0.70–0.80` può essere necessario, come indicato nel docstring di `defense.py`.

---

## Dove viene attivata

In `main.py`, la difesa è opzionale e si attiva con il flag `--defense`:

```
python main.py --defense
```

Corrisponde alla **Phase 2b** del flusso dell'esperimento, tra l'iniezione dei veleni e il training:

```
Phase 1  →  generazione veleni (attaccante)
Phase 2  →  iniezione nel training set
Phase 2b →  [se --defense] rimozione veleni prima del training   ← qui
Phase 3  →  training del modello vittima
Phase 4  →  valutazione attacco/difesa
```

---

## Perché funziona

La difesa **riusa il feature extractor dell'attaccante** (ResNet18 frozen). Questo è possibile perché:

- Il feature-collision attack è progettato esplicitamente per far collassare le feature dei veleni verso il target in quello spazio.
- Senza difesa, questa collisione inganna il classificatore durante il training.
- Con la difesa, la stessa collisione rende i veleni **rilevabili**: la loro alta similarità col target li tradisce prima che il training inizi.

In sostanza, la forza dell'attacco (la vicinanza in feature space) diventa la sua debolezza.

---

## Output diagnostico

Durante l'esecuzione, la difesa stampa:

- Range e media delle cosine similarity dei sample sospetti
- Numero totale di sample ispezionati
- Numero di veleni flaggati e rimossi
- Numero di sample mantenuti

Questi dati permettono di calibrare la soglia se i veleni non vengono rilevati.
