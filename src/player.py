# class to handle player logic
from src.decision import Decision
from src.game_state import GameState


class Player:
    def __init__(self, player_index: int):
        self.player_index = player_index

    def make_decision(self, game_state: GameState):
        pass

    def get_all_legal_decisions(self, game_state: GameState) -> list[Decision]:
        pass