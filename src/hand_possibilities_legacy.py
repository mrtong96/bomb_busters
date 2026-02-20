import functools
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import time

# wires are 10x rank + color, for example 4 yellow = 41
# can represent all of these as int8 arrays under the hood

class WireColor(Enum):
    BLUE = 0
    YELLOW = 1
    RED = 5

UNKNOWN = 0

@dataclass
class HandPossibilities:
    """
    Use this to store a bunch of pre-computed possible hands for the blue wires
    """
    num_cards: int
    hand_matrix: np.array
    weight_vector: np.array

def _get_hand_combinations(cards: int, min_card, card_limits: defaultdict) -> list[list[int]]:
    """
    :param cards: how many cards are left
    :param min_card: the minimum card value to start populating with
    :param card_limits: card limits as a dict
    :return:
    """
    # out of cards
    if cards == 0:
        return [[]]

    results = []

    for card in card_limits:
        # card is too small
        if card < min_card:
            continue
        # if we can play the card
        elif card_limits[card] > 0:
            card_limits[card] -= 1
            sub_results = _get_hand_combinations(cards - 1, card, card_limits)
            sub_results = [[card] + cur_result for cur_result in sub_results]
            results.extend(sub_results)
            card_limits[card] += 1

    return results

def get_possible_hands(num_cards: int, card_limits: defaultdict) -> HandPossibilities:
    """
    Get all the possible hands
    TODO: figure out if implementing card constraints here makes sense

    :param num_cards: number of cards
    :param card_limits: limits of the number of possible cards per int8 value. 0 is unknown
    :return: all the possible hands with their weighted probabilities
    """
    hand_combinations = _get_hand_combinations(num_cards, min_card=0, card_limits=card_limits)
    hand_combinations = np.array(hand_combinations)

    count_to_weight = {1: 4, 2: 6, 3: 4, 4: 1}
    weights = []
    for combo in hand_combinations:
        counts = Counter(combo).values()
        weight = 1
        for count in counts:
            weight *= count_to_weight[count]
        weights.append(weight)
    weights = np.array(weights)

    return HandPossibilities(num_cards=num_cards, hand_matrix=hand_combinations, weight_vector=weights)

@functools.cache
def get_blue_wire_hand_possibilities(num_cards: int) -> HandPossibilities:
    """
    Basis of forming
    :param num_cards: how many cards to generate for
    :return: the cached hand possibilities
    """
    limits_dict = defaultdict(int)
    for i in range(12):
        limits_dict[10 * (i + 1)] = 4
    return get_possible_hands(num_cards, limits_dict)

def main():
    t0 = time.time()
    for num_cards in range(1, 11):
        possible_hands = get_blue_wire_hand_possibilities(num_cards)
        print(possible_hands.hand_matrix.shape, np.sum(possible_hands.weight_vector))
    print(time.time() - t0)

if __name__ == '__main__':
    main()




