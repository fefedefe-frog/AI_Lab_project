# Tipi di mosse possibili
ACTIONS = ["HIT", "STAND", "DOUBLE", "SPLIT"]

# Tabella hard totals: (player total, dealer upcard) -> mossa
strategy_hard = {
    8:  {2: "HIT", 3: "HIT", 4: "HIT", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    9:  {2: "HIT", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    10: {2: "DOUBLE", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "DOUBLE", 8: "DOUBLE", 9: "DOUBLE", 10: "HIT", 11: "HIT"},
    11: {2: "DOUBLE", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "DOUBLE", 8: "DOUBLE", 9: "DOUBLE", 10: "DOUBLE", 11: "HIT"},
    12: {2: "HIT", 3: "HIT", 4: "STAND", 5: "STAND", 6: "STAND", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    13: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    14: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    15: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    16: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    17: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "STAND", 8: "STAND", 9: "STAND", 10: "STAND", 11: "STAND"},
}

# Tabella soft totals (es. A+6 = 17)
strategy_soft = {
    13: {2: "HIT", 3: "HIT", 4: "HIT", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    14: {2: "HIT", 3: "HIT", 4: "HIT", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    15: {2: "HIT", 3: "HIT", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    16: {2: "HIT", 3: "HIT", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    17: {2: "HIT", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    18: {2: "STAND", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "STAND", 8: "STAND", 9: "HIT", 10: "HIT", 11: "HIT"},
    19: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "STAND", 8: "STAND", 9: "STAND", 10: "STAND", 11: "STAND"},
    20: {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "STAND", 8: "STAND", 9: "STAND", 10: "STAND", 11: "STAND"},
}

# Tabella per coppie
strategy_pairs = {
    "A": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT", 8: "SPLIT", 9: "SPLIT", 10: "SPLIT", 11: "SPLIT"},
    "10": {2: "STAND", 3: "STAND", 4: "STAND", 5: "STAND", 6: "STAND", 7: "STAND", 8: "STAND", 9: "STAND", 10: "STAND", 11: "STAND"},
    "9": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "STAND", 8: "SPLIT", 9: "SPLIT", 10: "STAND", 11: "STAND"},
    "8": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT", 8: "SPLIT", 9: "SPLIT", 10: "SPLIT", 11: "SPLIT"},
    "7": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    "6": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    "5": {2: "DOUBLE", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE", 7: "DOUBLE", 8: "DOUBLE", 9: "DOUBLE", 10: "HIT", 11: "HIT"},
    "4": {2: "HIT", 3: "HIT", 4: "HIT", 5: "SPLIT", 6: "SPLIT", 7: "HIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    "3": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
    "2": {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT", 8: "HIT", 9: "HIT", 10: "HIT", 11: "HIT"},
}

def suggerisci_mossa(carte_player: list[str], carta_dealer: str) -> str:
    dealer_val = converti_valore(carta_dealer)
    totale, is_soft = calcola_totale_soft(carte_player)
    print("dealer_val: " + str(dealer_val) + "\n")
    print("is_soft: " +  str(is_soft) + "\n")
    print("totale: " + str(totale) + "\n")

    if len(carte_player) == 2 and converti_valore(carte_player[0]) == converti_valore(carte_player[1]):
      if(carte_player[0] in ["J", "Q", "K"]):
          carte_player[0] = "10"
      # Caso coppia
      return strategy_pairs.get(carte_player[0], {}).get(dealer_val, "HIT")

    if is_soft and totale <= 20:
      # Caso soft
      return strategy_soft.get(totale, {}).get(dealer_val, "HIT")

    # Caso hard
    return strategy_hard.get(totale, {}).get(dealer_val, "HIT")


def converti_valore(carta: str) -> int:
    if carta == "A":
        return 11
    elif carta in ["K", "Q", "J"]:
        return 10
    else:
        return int(carta)
    

def calcola_totale_soft(carte: list[str]) -> tuple[int, bool]:
    valori = []
    num_assi = 0
    for c in carte:
        if c == "A":
            num_assi += 1
            valori.append(11)
        elif c in ["K", "Q", "J"]:
            valori.append(10)
        else:
            valori.append(int(c))

    totale = sum(valori)
    assi_iniziali = num_assi
    while totale > 21 and num_assi > 0:
        totale -= 10 # ora l'asso vale 1
        num_assi -= 1

    # È soft se almeno un asso vale ancora 11
    is_soft = (assi_iniziali - num_assi) < assi_iniziali
    return totale, is_soft


if __name__ == "__main__":
    print("=== TEST 1 ===")
    mossa = suggerisci_mossa(["A", "6"], "5")
    print("Mossa consigliata:", mossa)  # DOUBLE

    print("=== TEST 2 ===")
    mossa2 = suggerisci_mossa(["9", "9"], "7")
    print("Mossa consigliata:", mossa2)  # STAND

    print("=== TEST 3 ===")
    mossa3 = suggerisci_mossa(["5", "3"], "6")
    print("Mossa consigliata:", mossa3)  # DOUBLE

    print("=== TEST 4 ===")
    mossa3 = suggerisci_mossa(["Q", "Q"], "Q")
    print("Mossa consigliata:", mossa3)  # STAND

    print("=== TEST 5 ===")
    mossa3 = suggerisci_mossa(["A", "3"], "6")
    print("Mossa consigliata:", mossa3)  # DOUBLE

    print("=== TEST 6 ===")
    mossa3 = suggerisci_mossa(["2", "3"], "6")
    print("Mossa consigliata:", mossa3)  # DOUBLE

    print("=== TEST 7 ===")
    mossa3 = suggerisci_mossa(["J", "10"], "6")
    print("Mossa consigliata:", mossa3)  # STAND