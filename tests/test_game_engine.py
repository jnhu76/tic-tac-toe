# -*- coding: utf-8 -*-
"""
GameEngine、Opponent 协议和增强训练集成测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.core.game import GameEngine, BOARD_SIZE, BOARD_ROWS, BOARD_COLS
from src.ai.train import NeuralNetwork, board_to_input, TicTacToeGame
from src.ai.opponents import (
    RandomOpponent,
    RuleBasedOpponent,
    NeuralNetOpponent,
    Opponent,
)


# ======================== GameEngine tests ========================


class TestGameEngine:
    def test_init(self):
        g = GameEngine()
        assert len(g.board) == BOARD_SIZE
        assert all(c is None for c in g.board)
        assert g.current_player == 0

    def test_get_symbol(self):
        g = GameEngine()
        assert g.get_symbol(0) == "X"
        assert g.get_symbol(1) == "O"

    def test_make_move_valid(self):
        g = GameEngine()
        assert g.make_move(4) is True
        assert g.board[4] == "X"
        assert g.current_player == 1

    def test_make_move_invalid_occupied(self):
        g = GameEngine()
        g.make_move(4)
        assert g.make_move(4) is False
        assert g.board[4] == "X"

    def test_make_move_invalid_out_of_range(self):
        g = GameEngine()
        assert g.make_move(-1) is False
        assert g.make_move(9) is False

    def test_player_switching(self):
        g = GameEngine()
        g.make_move(0)
        assert g.current_player == 1
        g.make_move(1)
        assert g.current_player == 0

    def test_get_board_2d(self):
        g = GameEngine()
        g.make_move(4)
        b2d = g.get_board_2d()
        assert len(b2d) == BOARD_ROWS
        assert len(b2d[0]) == BOARD_COLS
        assert b2d[1][1] == "X"
        assert b2d[0][0] is None

    def test_get_legal_moves_empty(self):
        g = GameEngine()
        assert len(g.get_legal_moves()) == BOARD_SIZE

    def test_get_legal_moves_partial(self):
        g = GameEngine()
        g.make_move(0)
        g.make_move(4)
        moves = g.get_legal_moves()
        assert len(moves) == 7
        assert 0 not in moves
        assert 4 not in moves

    def test_get_available_moves_alias(self):
        g = GameEngine()
        g.make_move(0)
        assert g.get_available_moves() == g.get_legal_moves()

    def test_reset(self):
        g = GameEngine()
        g.make_move(0)
        g.make_move(4)
        g.reset()
        assert all(c is None for c in g.board)
        assert g.current_player == 0

    def test_check_winner_row(self):
        g = GameEngine()
        g.board = ["X", "X", "X", None, None, None, None, None, None]
        assert g.check_winner() == "X"

    def test_check_winner_col(self):
        g = GameEngine()
        g.board = ["O", None, None, "O", None, None, "O", None, None]
        assert g.check_winner() == "O"

    def test_check_winner_diag(self):
        g = GameEngine()
        g.board = ["X", None, None, None, "X", None, None, None, "X"]
        assert g.check_winner() == "X"

    def test_check_winner_anti_diag(self):
        g = GameEngine()
        g.board = [None, None, "O", None, "O", None, "O", None, None]
        assert g.check_winner() == "O"

    def test_no_winner(self):
        g = GameEngine()
        assert g.check_winner() is None

    def test_is_game_over_win(self):
        g = GameEngine()
        g.board = ["X", "X", "X", None, None, None, None, None, None]
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "X"

    def test_is_game_over_tie(self):
        g = GameEngine()
        g.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "T"

    def test_is_game_over_not_yet(self):
        g = GameEngine()
        over, winner = g.is_game_over()
        assert over is False
        assert winner is None

    def test_clone(self):
        g = GameEngine()
        g.make_move(0)
        g.make_move(4)
        g2 = g.clone()
        assert g2.board[0] == "X"
        assert g2.board[4] == "O"
        g2.make_move(8)
        assert g.board[8] is None  # original unchanged

    def test_full_game_x_wins(self):
        g = GameEngine()
        # X: 0, 1, 2 (top row)
        g.make_move(0)  # X
        g.make_move(3)  # O
        g.make_move(1)  # X
        g.make_move(4)  # O
        g.make_move(2)  # X wins
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "X"

    def test_full_game_tie(self):
        g = GameEngine()
        moves = [0, 4, 8, 2, 6, 3, 5, 7, 1]  # alternating X/O
        for m in moves:
            g.make_move(m)
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "T"


# ======================== TicTacToeGame inherits GameEngine ========================


class TestTicTacToeGameInheritance:
    def test_is_game_engine(self):
        g = TicTacToeGame()
        assert isinstance(g, GameEngine)

    def test_has_all_methods(self):
        g = TicTacToeGame()
        assert hasattr(g, "make_move")
        assert hasattr(g, "check_winner")
        assert hasattr(g, "get_legal_moves")
        assert hasattr(g, "get_available_moves")
        assert hasattr(g, "clone")
        assert hasattr(g, "get_board_2d")
        assert hasattr(g, "is_game_over")


# ======================== Opponent protocol tests ========================


class TestOpponentProtocol:
    def test_random_opponent_is_opponent(self):
        assert isinstance(RandomOpponent(), Opponent)

    def test_rule_opponent_is_opponent(self):
        assert isinstance(RuleBasedOpponent(), Opponent)

    def test_neural_net_opponent_is_opponent(self):
        nn = NeuralNetwork()
        assert isinstance(NeuralNetOpponent(nn), Opponent)

    def test_random_opponent_returns_legal_move(self):
        g = GameEngine()
        g.make_move(0)
        opp = RandomOpponent()
        move = opp.get_move(g)
        assert move in g.get_legal_moves()

    def test_rule_opponent_blocks_win(self):
        g = GameEngine()
        # X at 0, 1 — O must block at 2
        g.board = ["X", "X", None, None, None, None, None, None, None]
        g.current_player = 1  # O's turn
        opp = RuleBasedOpponent()
        move = opp.get_move(g)
        assert move == 2

    def test_rule_opponent_takes_win(self):
        g = GameEngine()
        # O at 3, 4 — O must take 5 to win
        g.board = [None, None, None, "O", "O", None, None, None, None]
        g.current_player = 1  # O's turn
        opp = RuleBasedOpponent()
        move = opp.get_move(g)
        assert move == 5

    def test_rule_opponent_takes_center(self):
        g = GameEngine()
        g.board = ["X", None, None, None, None, None, None, None, None]
        g.current_player = 1  # O's turn
        opp = RuleBasedOpponent()
        move = opp.get_move(g)
        assert move == 4

    def test_neural_net_opponent_returns_legal_move(self):
        g = GameEngine()
        g.make_move(0)
        nn = NeuralNetwork()
        opp = NeuralNetOpponent(nn)
        move = opp.get_move(g)
        assert move in g.get_legal_moves()

    def test_neural_net_opponent_with_epsilon(self):
        g = GameEngine()
        nn = NeuralNetwork()
        opp = NeuralNetOpponent(nn, epsilon=1.0)
        move = opp.get_move(g)
        assert move in g.get_legal_moves()


# ======================== NeuralNetwork.copy tests ========================


class TestNeuralNetworkCopy:
    def test_copy_returns_new_instance(self):
        nn = NeuralNetwork()
        nn2 = nn.copy()
        assert nn is not nn2

    def test_copy_has_same_weights(self):
        nn = NeuralNetwork()
        nn2 = nn.copy()
        assert np.allclose(nn.weights_ih, nn2.weights_ih)
        assert np.allclose(nn.weights_ho, nn2.weights_ho)
        assert np.allclose(nn.biases_h, nn2.biases_h)
        assert np.allclose(nn.biases_o, nn2.biases_o)

    def test_copy_is_independent(self):
        nn = NeuralNetwork()
        nn2 = nn.copy()
        nn2.weights_ih[0, 0] = 999
        assert nn.weights_ih[0, 0] != 999


# ======================== Enhanced training integration tests ========================


class TestEnhancedTraining:
    def test_import_all_functions(self):
        from src.ai.train_enhanced import (
            FirstMoverConfig,
            train_with_checkpoint_and_firstmover,
            evaluate_with_firstmover,
            comprehensive_evaluation,
            random_opponent_move,
            rule_based_opponent_move,
            self_play_move,
        )

    def test_enhanced_training_smoke(self):
        from src.ai.train_enhanced import (
            FirstMoverConfig,
            train_with_checkpoint_and_firstmover,
        )
        from src.core.checkpoint import CheckpointManager
        import tempfile

        nn = NeuralNetwork()
        cm = CheckpointManager(checkpoint_dir=tempfile.mkdtemp(), verbose=False)
        cfg = FirstMoverConfig(ai_first_ratio=0.5)

        nn = train_with_checkpoint_and_firstmover(
            nn,
            total_episodes=200,
            learning_rate=0.1,
            checkpoint_manager=cm,
            firstmover_config=cfg,
            opponent_type="random",
            eval_interval=100,
            log_interval=100,
            num_eval_games=20,
            model_save_path="/tmp/test_model_smoke.pkl",
        )
        assert nn is not None

    def test_evaluate_with_firstmover(self):
        from src.ai.train_enhanced import evaluate_with_firstmover

        nn = NeuralNetwork()
        opp = RuleBasedOpponent()
        result = evaluate_with_firstmover(nn, opp, num_games=20, ai_first=True)
        assert "win_rate" in result
        assert "draw_rate" in result
        assert "loss_rate" in result
        assert abs(result["win_rate"] + result["draw_rate"] + result["loss_rate"] - 1.0) < 1e-6

    def test_evaluate_with_callable(self):
        from src.ai.train_enhanced import evaluate_with_firstmover

        nn = NeuralNetwork()

        def simple_opponent(game):
            return game.get_legal_moves()[0]

        result = evaluate_with_firstmover(nn, simple_opponent, num_games=10, ai_first=True)
        assert "win_rate" in result

    def test_comprehensive_evaluation(self):
        from src.ai.train_enhanced import comprehensive_evaluation

        nn = NeuralNetwork()
        results = comprehensive_evaluation(nn, num_games=20)
        assert "random_first" in results
        assert "rule_first" in results
        assert "summary" in results
