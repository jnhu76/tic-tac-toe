import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.ai.train import (
    BOARD_SIZE,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    NeuralNetwork,
    TicTacToeGame,
    board_to_input,
    get_computer_move,
    get_random_move,
    learn_from_game,
    load_model,
    play_random_game,
    save_model,
)
from src.core.common import BOARD_COL, BOARD_ROW, GameBoard

# ======================== GameBoard tests ========================


class TestGameBoard:
    def test_init(self):
        b = GameBoard()
        assert b.col == 3
        assert b.row == 3
        assert b.board_size == 9
        assert b.current_player == 0
        for row in b.board:
            for cell in row:
                assert cell is None

    def test_reset(self):
        b = GameBoard()
        b.board[0][0] = "X"
        b.current_player = 1
        b.reset_board()
        for row in b.board:
            for cell in row:
                assert cell is None
        assert b.current_player == 0


# ======================== TicTacToeGame tests ========================


class TestTicTacToeGame:
    def test_init(self):
        g = TicTacToeGame()
        assert len(g.board) == 9
        assert all(c is None for c in g.board)
        assert g.current_player == 0

    def test_get_symbol(self):
        g = TicTacToeGame()
        assert g.get_symbol(0) == "X"
        assert g.get_symbol(1) == "O"

    def test_make_move_valid(self):
        g = TicTacToeGame()
        assert g.make_move(4) is True
        assert g.board[4] == "X"
        assert g.current_player == 1

    def test_make_move_invalid_occupied(self):
        g = TicTacToeGame()
        g.make_move(4)
        assert g.make_move(4) is False
        assert g.board[4] == "X"

    def test_make_move_invalid_out_of_range(self):
        g = TicTacToeGame()
        assert g.make_move(-1) is False
        assert g.make_move(9) is False

    def test_player_switching(self):
        g = TicTacToeGame()
        g.make_move(0)
        assert g.current_player == 1
        g.make_move(1)
        assert g.current_player == 0

    def test_get_board_2d(self):
        g = TicTacToeGame()
        g.make_move(4)
        b2d = g.get_board_2d()
        assert b2d[1][1] == "X"
        assert b2d[0][0] is None

    def test_get_available_moves_empty(self):
        g = TicTacToeGame()
        assert len(g.get_available_moves()) == 9

    def test_get_available_moves_partial(self):
        g = TicTacToeGame()
        g.make_move(0)
        g.make_move(4)
        moves = g.get_available_moves()
        assert len(moves) == 7
        assert 0 not in moves
        assert 4 not in moves

    def test_reset(self):
        g = TicTacToeGame()
        g.make_move(0)
        g.make_move(4)
        g.reset()
        assert all(c is None for c in g.board)
        assert g.current_player == 0

    def test_check_winner_row(self):
        g = TicTacToeGame()
        g.board[0] = "X"
        g.board[1] = "X"
        g.board[2] = "X"
        assert g.check_winner() == "X"

    def test_check_winner_col(self):
        g = TicTacToeGame()
        g.board[0] = "O"
        g.board[3] = "O"
        g.board[6] = "O"
        assert g.check_winner() == "O"

    def test_check_winner_diag(self):
        g = TicTacToeGame()
        g.board[0] = "X"
        g.board[4] = "X"
        g.board[8] = "X"
        assert g.check_winner() == "X"

    def test_check_winner_anti_diag(self):
        g = TicTacToeGame()
        g.board[2] = "O"
        g.board[4] = "O"
        g.board[6] = "O"
        assert g.check_winner() == "O"

    def test_no_winner(self):
        g = TicTacToeGame()
        assert g.check_winner() is None

    def test_no_winner_partial(self):
        g = TicTacToeGame()
        g.board[0] = "X"
        g.board[1] = "O"
        g.board[2] = "X"
        assert g.check_winner() is None

    def test_is_game_over_win(self):
        g = TicTacToeGame()
        g.board[0] = "X"
        g.board[1] = "X"
        g.board[2] = "X"
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "X"

    def test_is_game_over_tie(self):
        g = TicTacToeGame()
        g.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        over, winner = g.is_game_over()
        assert over is True
        assert winner == "T"

    def test_is_game_over_not_yet(self):
        g = TicTacToeGame()
        over, winner = g.is_game_over()
        assert over is False
        assert winner is None


# ======================== board_to_input tests ========================


class TestBoardToInput:
    def test_empty_board(self):
        board = [[None] * 3 for _ in range(3)]
        inp = board_to_input(board)
        assert inp.shape == (NN_INPUT_SIZE,)
        assert np.all(inp == 0)

    def test_x_encoding(self):
        board = [["X", None, None], [None, None, None], [None, None, None]]
        inp = board_to_input(board)
        assert inp[0] == 1
        assert inp[1] == 0
        assert inp[2] == 0
        assert inp[3] == 0

    def test_o_encoding(self):
        board = [[None, None, None], [None, "O", None], [None, None, None]]
        inp = board_to_input(board)
        pos = 1 * 3 + 1
        assert inp[pos * 2] == 0
        assert inp[pos * 2 + 1] == 1

    def test_mixed_board(self):
        board = [["X", "O", None], [None, "X", None], [None, None, "O"]]
        inp = board_to_input(board)
        assert inp[0] == 1
        assert inp[1] == 0
        assert inp[2] == 0
        assert inp[3] == 1
        assert inp[4] == 0
        assert inp[5] == 0


# ======================== NeuralNetwork tests ========================


class TestNeuralNetwork:
    def test_init_shapes(self):
        nn = NeuralNetwork()
        assert nn.weights_ih.shape == (NN_INPUT_SIZE, NN_HIDDEN_SIZE)
        assert nn.weights_ho.shape == (NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
        assert nn.biases_h.shape == (NN_HIDDEN_SIZE,)
        assert nn.biases_o.shape == (NN_OUTPUT_SIZE,)

    def test_forward_output_shape(self):
        nn = NeuralNetwork()
        inp = np.zeros(NN_INPUT_SIZE)
        out = nn.forward(inp)
        assert out.shape == (NN_OUTPUT_SIZE,)

    def test_forward_softmax_sums_to_one(self):
        nn = NeuralNetwork()
        inp = np.random.randn(NN_INPUT_SIZE)
        out = nn.forward(inp)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_forward_all_nonnegative(self):
        nn = NeuralNetwork()
        inp = np.random.randn(NN_INPUT_SIZE)
        out = nn.forward(inp)
        assert np.all(out >= 0)

    def test_backward_changes_weights(self):
        nn = NeuralNetwork()
        inp = np.zeros(NN_INPUT_SIZE)
        nn.forward(inp)
        w_before = nn.weights_ho.copy()
        target = np.zeros(NN_OUTPUT_SIZE)
        target[0] = 1.0
        nn.backward(target, 0.1, 1.0)
        assert not np.allclose(w_before, nn.weights_ho)

    def test_get_best_move_returns_valid(self):
        nn = NeuralNetwork()
        board = [[None] * 3 for _ in range(3)]
        inp = board_to_input(board)
        move = nn.get_best_move(inp, board)
        assert 0 <= move < 9

    def test_get_best_move_ignores_occupied(self):
        nn = NeuralNetwork()
        board = [["X", "X", "X"], ["X", "X", "X"], [None, None, None]]
        inp = board_to_input(board)
        move = nn.get_best_move(inp, board)
        assert move in [6, 7, 8]

    def test_softmax_uniform_on_equal(self):
        nn = NeuralNetwork()
        x = np.zeros(5)
        result = nn.softmax(x)
        assert np.allclose(result, np.ones(5) / 5)

    def test_softmax_large_values(self):
        nn = NeuralNetwork()
        x = np.array([1000.0, 1000.0, 1000.0])
        result = nn.softmax(x)
        assert np.allclose(result, np.ones(3) / 3)


# ======================== Training integration tests ========================


class TestTraining:
    def test_play_random_game_returns_valid(self):
        nn = NeuralNetwork()
        winner = play_random_game(nn)
        assert winner in ("X", "O", "T")

    def test_learn_from_game_modifies_weights(self):
        nn = NeuralNetwork()
        w_before = nn.weights_ho.copy()
        learn_from_game(nn, [0, 4, 1, 5, 2], 0, "X")
        assert not np.allclose(w_before, nn.weights_ho)

    def test_positive_reward_for_win(self):
        nn = NeuralNetwork()
        w_before = nn.weights_ho.copy()
        learn_from_game(nn, [0, 4, 1, 5, 2], 0, "X")
        assert not np.allclose(w_before, nn.weights_ho)

    def test_negative_reward_for_loss(self):
        nn = NeuralNetwork()
        w_before = nn.weights_ho.copy()
        learn_from_game(nn, [0, 4, 1, 5, 2], 1, "X")
        assert not np.allclose(w_before, nn.weights_ho)

    def test_get_random_move_valid(self):
        g = TicTacToeGame()
        move = get_random_move(g)
        assert 0 <= move < 9
        assert g.board[move] is None

    def test_get_computer_move_valid(self):
        nn = NeuralNetwork()
        g = TicTacToeGame()
        move = get_computer_move(g, nn)
        assert 0 <= move < 9

    def test_training_improves_winrate(self):
        nn = NeuralNetwork()
        for _ in range(2000):
            play_random_game(nn)
        wins = 0
        for _ in range(200):
            g = TicTacToeGame()
            while True:
                if g.current_player == 0:
                    move = get_random_move(g)
                else:
                    move = get_computer_move(g, nn)
                g.board[move] = g.get_symbol(g.current_player)
                g.current_player = 1 - g.current_player
                over, winner = g.is_game_over()
                if over:
                    if winner == "O":
                        wins += 1
                    break
        assert (
            wins > 80
        ), f"AI should win >40% after 2000 training games, got {wins}/200"


# ======================== Save/Load tests ========================


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        nn = NeuralNetwork()
        path = str(tmp_path / "model.pkl")
        save_model(nn, path)
        loaded = load_model(path)
        assert np.allclose(loaded.weights_ih, nn.weights_ih)
        assert np.allclose(loaded.weights_ho, nn.weights_ho)
        assert np.allclose(loaded.biases_h, nn.biases_h)
        assert np.allclose(loaded.biases_o, nn.biases_o)
