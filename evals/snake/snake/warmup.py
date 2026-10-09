"""Shared warmup for the snake clients: a few decisions on a throwaway game."""


def warm_up(policy, game, steps=6):
    """A won or dead game has no moves left; stop before asking the policy
    again on an empty move list."""
    for _ in range(steps):
        if not game.alive or game.won:
            break
        decision = policy.decide(game)
        game.step(decision.executed)
