from src.hand_possibilities_legacy import get_blue_wire_hand_possibilities
import numpy as np

def main():
    ten_card_possibilities = get_blue_wire_hand_possibilities(10)

    probability_weights = np.zeros((10, 12), dtype=np.int64)

    for weight, row in zip(ten_card_possibilities.weight_vector, ten_card_possibilities.hand_matrix):
        for index, el in enumerate(row):
            probability_weights[index][int((el - 10) / 10)] += weight

    probability_weights = probability_weights.astype(np.float64)
    print(np.sum(probability_weights))

    for i in range(10):
        total_weight = np.sum(probability_weights[i])
        probability_weights[i] /= total_weight

    shannon_entropy = 0
    for i in range(10):
        for j in range(12):
            prob = probability_weights[i, j]
            if prob == 0:
                continue
            shannon_entropy += - prob * np.log2(prob)
    print(shannon_entropy)

if __name__ == '__main__':
    main()