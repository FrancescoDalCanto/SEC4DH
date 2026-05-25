# Progetto SEC4DH
### Clean-Label Data Poisoning Attack e Defense su BreastMNIST

---

## Cos'è il progetto

SEC4DH è un esperimento che dimostra quanto possa essere vulnerabile un classificatore di immagini mediche agli attacchi di data poisoning, e come sia possibile difendersi. Il contesto è la classificazione binaria di ecografie mammarie tratte dal dataset BreastMNIST, dove ogni immagine è etichettata come **Maligna (0)** o **Normale (1)**. L'obiettivo è simulare uno scenario realistico in cui un attaccante riesce a manipolare il comportamento di un modello di machine learning corrompendo i dati di addestramento, senza però che le immagini alterate sembrino sospette a occhio nudo.

---

## Il dataset e il modello vittima

BreastMNIST è un dataset pubblico di ecografie mammarie in scala di grigi. Le immagini vengono convertite a RGB e ridimensionate a 224×224 pixel per essere compatibili con ResNet18. Il modello vittima è un ResNet18 pre-addestrato su ImageNet e adattato tramite **transfer learning**: il backbone rimane congelato e si allena solo il classificatore binario finale per 10 epoche. Questo setup è tipico in ambito medico, dove i dati etichettati sono pochi e si sfruttano reti già addestrate su grandi dataset generici.

---

## Il cuore dell'attacco

L'attacco si chiama **Clean-Label Poisoning** e si basa su un'idea semplice ma efficace: creare immagini che a occhio sembrano perfettamente normali (e vengono etichettate come tali), ma che nel feature space del modello si comportano come l'immagine maligna bersaglio.

Il meccanismo centrale è la **feature collision**: si prende un'immagine normale come base e la si modifica con una piccola perturbazione (al massimo 8/255 per pixel, invisibile all'occhio umano) ottimizzata tramite Adam per 800 passi, in modo che la sua rappresentazione interna nel ResNet18 si avvicini il più possibile a quella dell'immagine maligna. Il modello vede queste immagini come Normali durante il training, ma ha imparato ad associare quella rappresentazione alla classe sbagliata: il risultato è che quando arriva una vera immagine maligna, il modello la classifica come Normale.

---

## Il cuore della difesa

La difesa sfrutta esattamente la proprietà che rende efficace l'attacco: i veleni sono artificialmente vicini al bersaglio nello spazio delle feature. Prima del training, si calcola la **cosine similarity** tra ogni campione etichettato Normale e l'immagine bersaglio nel feature space di ResNet18. I campioni la cui similarità supera la soglia **0.70** vengono rimossi dal dataset. Le immagini normali legittime sono naturalmente lontane da un'immagine maligna in quello spazio; i veleni invece, costruiti apposta per essere vicini al bersaglio, vengono individuati ed eliminati prima che il modello li veda.

---

## Come funziona l'esperimento

L'esperimento si divide in cinque fasi:

1. **Poison generation** — l'attaccante genera 20 immagini veleno ottimizzando la feature collision sulle 20 immagini normali più simili al bersaglio.
2. **Poison deployment** — le immagini vengono inserite nel training set con etichetta Normale; se la difesa è attiva (`--defense`), i veleni vengono rimossi prima di procedere.
3. **Victim training** — il modello vittima viene addestrato per 10 epoche sul dataset (eventualmente sanificato).
4. **Evaluation** — si valuta quanti dei 25 campioni maligni di test vengono classificati erroneamente come Normali.
5. **Visualization** — si salvano immagini di esempio e un grafico riassuntivo nella cartella `imgs/`.

---

## Cosa si misura

L'efficacia si misura su 25 campioni maligni di test: si conta quanti vengono classificati erroneamente come Normali (**attack success ratio**). Si riporta anche la predizione sull'immagine bersaglio specifica e la confidenza media sulla classe Normale. Un attacco riuscito produce un alto tasso di misclassificazione; la difesa lo riporta vicino a zero.
