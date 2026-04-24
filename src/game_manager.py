# class to keep track of the state of the game state and the players
# For now just focus on the basic gameplay of guess->indicate and worry about
# items later
from typing import Optional

from src.decision import DualCutDecision, AskerResponseDecision, AskeeResponseDecision
from src.game_state import GameState
from src.player import Player


class GameManager:
    def __init__(
            self,
            game_state: Optional[GameState] = None,
            num_players: int=5,
            yellow_wires: Optional[set[int]] = None,
            yellow_wires_in_hands: int = 0,
            red_wires: Optional[set[int]] = None,
            red_wires_in_hands: int = 0,
            # how many wrong moves you can make before things blow up
            total_health: int = 5,
            # who's next to move. Default is the first player (0)
            player_to_move: int = 0,
    ):
        if game_state is None:
            self.game_state = GameState(
                num_players=num_players,
                yellow_wires=yellow_wires,
                yellow_wires_in_hands=yellow_wires_in_hands,
                red_wires=red_wires,
                red_wires_in_hands=red_wires_in_hands,
                total_health=total_health,
                player_to_move=player_to_move,
            )
        else:
            self.game_state = game_state

        self.players = [Player(player_index = i) for i in range(num_players)]

    def process_turn(self):
        # the first player makes a decision, record it
        cur_decision = self.players[self.game_state.player_to_move].make_decision(self.game_state)
        self.game_state.update_constraints_from_decision(cur_decision)
        self.game_state.turns.append([cur_decision])

        while not cur_decision.is_turn_ending_decision:
            if isinstance(cur_decision, DualCutDecision):
                askee_index = cur_decision.askee_player_index
                cur_decision = self.players[askee_index].make_decision(self.game_state)
            elif isinstance(cur_decision, AskeeResponseDecision):
                asker_index = self.game_state.player_to_move
                cur_decision = self.players[asker_index].make_decision(self.game_state)
            else:
                raise NotImplementedError(f"Not implemented yet to continue from a decision {type(cur_decision)}")

            self.game_state.most_recent_turn.append(cur_decision)

            if len(self.game_state.most_recent_turn) > 3:
                raise RuntimeError("turns should not take more than 3 decisions")

        # end of turn, increment the player turn
        self.game_state.increment_player_to_move()

    # def process_player_decision(self):
    #     pass