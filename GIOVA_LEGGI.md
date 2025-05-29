# PER FARE DATASET CNN:  
## da ./Progetto fai `python -m cnn_dataset_maker.generate_dataset <numero di immagini per foto>`  
# PER FARE DATASET YOLO:  
## da ./Progetto fai `python -m YOLO_dataset_maker.main <numero di immagini per foto> -r` il "-r" serve per generare le foto con rotazioni random, se non lo metti le genera tutte dritte  



# RICORDA: avvia sempre gli script dal terminale normale non da quello di vscode che si bugga, dovrei aver sistemato per eventuali errori ma non so sicuro non ho controllato  

## CNN:  
per la cnn la puoi avviare normalmente, non modificare o eliminare i print che sono tutti incastrati fra di loro,  
se ne vuoi agginugere basta che siano dopo il print("fatto") perché quelli prima vengono sovrascritti e il cursore  
viene spostato più volte su e giù tra le righe per creare l'effetto della console con le varie barre, senza avere  
un print infinito di robba  

### Da fare:  
fai i vari test seguendo i TODO nei file della cnn, e poi magari mettiti a vedere se riesci a ottimizzarla meglio  
poi se ti va prova a testare un nuovo modello di yolo sulle immagini generate da yolo_dataset_maker almeno in caso  
usiamo quello che è tutta roba che ha spiegato il prof co openCV2