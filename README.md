Overall goals
* Figure out how to maximize win percentage chance in increasingly complicated scenarios
* Be able to make decisions conditioned on the fact that other players make suboptimal and non-random decisions
* Make sure that all the calculations can be run in real-time ish (<1 min or less per decision) from a laptop

Random notes/thoughts
* The blue wires really behave like cards with no suit.
  * There are only 12 ranks instead of 13 but it's close enough
  * Red/Yellow wires slightly break this assumption
* It is unclear what a good winning heuristic is. Possible factors include
  * Reducing entropy
  * Maximizing the probability of a guaranteed safe move
  * Minimizing the probability of triggering an instant loss condition
  * Considering the state of the game (how much health you have/could have)
* There aren't that many hands that each individual could have
  * ncr(21, 10) - 12 * ncr(16, 5) + ncr(12, 2) = 300_366 for 10 cards, only blue wires
  * Worst case is to 2x the number of combinations for the presensce/abscence of each red/yellow
  * Point is that there shouldn't be that many possible hand combinations if you take each player independently
  * Joint probability calculations are likely out of the question, assume independence for base probabilities of hand position/freq for initial seeding
  * At least for the first draft enumerating through all possible hands to get weighted priors is reasonable
  * Can even easily compute weights by looking at rank frequencies for each possible hand
* Probably should copy the minesweeper solvers
  * Do joint probabilities with nice independent assumptions with a large state space
  * With a small state space try to estimate priors and brute force the solution space
  * For estimating priors with small state space most of the tiles are known
    * You can probably do some conditional probability with log(p(making_moves | hand)) for each player
    * Might need a temperature parameter to factor in "human randomness"


