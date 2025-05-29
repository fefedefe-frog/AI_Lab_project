# ISTRUZIONI PER USARE I VARI SCRIPT  
Qui sotto sono scritte le varie istruzioni su come usare i vari script per utilizzare i vari modelli  
e/o allenarli, e generare i loro rispettivi dataset.


## CNN
### **Dataset:** 
Per generare un dataset randomico per la cnn, stando nella stessa cartella del progetto, basta eseguire questo comando:  
    ``python -m cnn_dataset_maker.generate_dataset <numero di immagini per foto>``  
Di default lo script salva il dataset nella cartella dove si trova, se si vuole salvare il dataset in un altra cartella  
basta aggiungere l'opzione ``-o`` e specificare il percorso della cartella dove si vuole salvare il dataset    

### **Train/Test:**
Per avviare i loop di train e test della cnn, stando nella cartella del progetto, basta eseguire questo comando:  
    ``python -m CNN_Double_head.main <path_del_dataset> <path_del_csv> <numero di epochs>``  
Questo avviera il main script della cnn che procederà ad allenarla e salvarla sottoforma di pesi, in più vengono salvati  
anche i risultati dell'allenamento sottoforma di immagini, due per le confusion matrix e una per loss e accuracy.  
questo script ha due opzioni aggiuntive: ``-b`` se si vuole specificare il numero della grandezza del batch per i due  
dataloader (di default 32), e ``-o`` se si vuole specificare un path dove salvare i risultati (di default nella stessa  
cartella dello script)

### **Utilizzo per riconoscimento:**
Da fare, ma per ora basta aprire e modificare il file predict_card_image.py  


## YOLO
### **Dataset:** 
Se si vuole generare un dataset per allenare il modello di yolo basta eseguire il seguente comando, stando nella cartella  
del progetto:  
    `python -m YOLO_dataset_maker.main <immagini_per_foto> -r`  
Questo comando andrà a generare delle immagini di carte da gioco, posizionate in modo randomico su uno degli sfondi disponibili,  
oltre a posizionarle randomicamente le ruoterà randomicamente rispetto al punto centrale della carte poichè è specificata  
l'opzione ``-r``, oltre a questa opzione esiste l'opzione ``-o`` che può essere usata per specificare un path preciso dove  
si vuole salvare il dataset, di default il dataset viene salvato nella stessa cartella dello script.  

### **Train/Test:**
Per avviare il train del modello di yolo basta usare questo comando, stando nella cartella del progetto:  
    ``python -m YOLO_card_detector.train_yolo <data_path> <epochs>``  
specificando il path del file data.yaml e il numero di epochs per cui lo si vuole allenare. Questo script ha anche altre  
opzioni disponibili quali: ``-i`` per specificare al modello quanto deve ridimensionare le immagini che riceve in input  
durante l'allenamento, ``-o`` per specificare il nome della cartella di output.

### **Utilizzo per riconoscimento:**
Per ora bisogna aprire e modificare il file detect_card_with_yolo.py