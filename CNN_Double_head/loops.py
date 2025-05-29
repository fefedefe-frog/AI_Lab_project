import torch
import sys


# IMPORTANTE: dato che la CNN ha due teste separate, e una impara più infretta dell'altra (i semi)
# si applicano questi weitght per ribilanciare l'apprendimento, è il metodo più semplice e veloce
# per ovviare al problema delle due teste
WEIGHT_BALANCER_SEME: float= 0.3
WEIGHT_BALANCER_NUMERO: float= 0.7

def training_loop(model, dataloader, metric_seme, metric_numero, loss_fn, optimizer, device) -> tuple[list, list, list, list]:
    # Array contenenti tutte le loss e le accuracy per poter fare il plot finale
    losses_seme: list = []
    losses_numero: list = []
    accuracy_seme: list = []
    accuracy_numero: list = []

    model.train()
    dataset_size = len(dataloader)

    # Reset delle metriche
    metric_seme.reset()
    metric_numero.reset()

    # Recupero il batch di dati dal disco
    first_print: bool= True
    for batch, (images, seme_labels, numero_labels) in enumerate(dataloader):

        images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)

        # Uso il modello per ottenere le predizioni delle classi
        pred_seme, pred_numero = model(images)

        # Calcolo l'errore usando la loss function
        loss_seme = loss_fn(pred_seme, seme_labels)
        loss_numero = loss_fn(pred_numero, numero_labels)

        # Aggiorno gli array per il plot finale
        losses_seme.append(loss_seme.item())
        losses_numero.append(loss_numero.item())

        loss = WEIGHT_BALANCER_SEME * loss_seme + WEIGHT_BALANCER_NUMERO * loss_numero

        # Eseguo il passaggio di backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggiorno le metriche
        metric_seme.update(pred_seme, seme_labels)
        metric_numero.update(pred_numero, numero_labels)

        # Stampa le statistiche ogni x batch(10 in questo caso)

        if batch % 10 == 0:

            if not first_print:
                sys.stdout.write('\033[F\033[K\033[F\033[K')
            else:
                first_print= False

            loss, current = loss.item(), (batch + 1) * len(images)
            acc_seme = metric_seme.compute()
            acc_numero = metric_numero.compute()

            print(f"Loss: {loss}, [{current:>5} / {dataset_size}]")
            print(f"Accuracy seme: {acc_seme} || {acc_seme * 100:.2f}%")
            print(f"Accuracy numero: {acc_numero} || {acc_numero * 100:.2f}%")

    # Stampo l'accuratezza a fine train
    acc_s = metric_seme.compute()
    acc_n = metric_numero.compute()

    print("\n== Final Training Accuracy ==")
    print(f"\t- Seme: {acc_s} || {acc_s * 100:.2f}%")
    print(f"\t- Numero: {acc_n} || {acc_n * 100:.2f}%")

    # Aggiorno l'array per il plot finale
    accuracy_seme.append(acc_s.item())
    accuracy_numero.append(acc_n.item())

    return losses_seme, losses_numero, accuracy_seme, accuracy_numero

def testing_loop(model, dataloader, metric_seme, metric_numero, loss_fn, device) -> tuple[dict[str:list], dict[str:list], list, list, list, list]:
    # Dict di Array per le Confusion Matrix
    all_labels: dict = {'seme': [], 'numero': []}
    all_preds: dict = {'seme': [], 'numero': []}

    # Array contenenti tutte le loss e le accuracy per poter fare il plot finale
    losses_seme: list = []
    losses_numero: list = []
    accuracy_seme: list = []
    accuracy_numero: list = []

    model.eval()

    metric_seme.reset()
    metric_numero.reset()
    with torch.no_grad():
        for images, seme_labels, numero_labels in dataloader:
            images, seme_labels, numero_labels = images.to(device), seme_labels.to(device), numero_labels.to(device)
            seme_logits, numero_logits = model(images)

            pred_seme = seme_logits.argmax(dim=1)
            pred_numero = numero_logits.argmax(dim=1)

            # Salvo il valori per le confusion matrix
            all_preds['seme'].append(pred_seme.cpu().numpy())
            all_labels['seme'].append(seme_labels.numpy())

            all_preds['numero'].append(pred_numero.cpu().numpy())
            all_labels['numero'].append(numero_labels.numpy())

            # Aggiorno le metriche
            metric_seme.update(pred_seme, seme_labels)
            metric_numero.update(pred_numero, numero_labels)

            # Aggiorno le loss
            loss_s = loss_fn(seme_logits, seme_labels)
            loss_n = loss_fn(numero_logits, numero_labels)

            losses_seme.append(loss_s.item())
            losses_numero.append(loss_n.item())

    # Stampo l'accuratezza a fine loop
    acc_s = metric_seme.compute()
    acc_n = metric_numero.compute()
    print(f"\n\nFinal Testing Accuracy \n\t- Seme: {acc_s} || {acc_s * 100:.2f}%\n\t- Numero: {acc_n} || {acc_n * 100:.2f}%")

    accuracy_seme.append(acc_s.item())
    accuracy_numero.append(acc_n.item())

    return all_labels, all_preds, losses_seme, losses_numero, accuracy_seme, accuracy_numero