import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import tempfile

import pytest

from src.ai.train import NeuralNetwork
from src.ai.train_enhanced import FirstMoverConfig
from src.core.checkpoint import (
    CheckpointManager,
    GameState,
    TrainingState,
    create_checkpoint_manager,
)


class TestCheckpointManager:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.cm = CheckpointManager(
            checkpoint_dir=self.test_dir,
            max_checkpoints=5,
            auto_save_interval=1000,
            verbose=False,
        )
        self.nn = NeuralNetwork()

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        assert Path(self.test_dir).exists()
        assert (Path(self.test_dir) / "training").exists()
        assert (Path(self.test_dir) / "game").exists()

    def test_save_and_load_training(self):
        state = TrainingState(
            episode=100,
            epsilon=0.5,
            best_win_rate_rule=0.6,
            best_win_rate_random=0.7,
            learning_rate=0.001,
            total_episodes=1000,
            phase=0,
            metrics_history=[],
            timestamp="2026-04-13 12:00:00",
        )
        fp = self.cm.save_training_checkpoint(self.nn, state)
        assert Path(fp).exists()

        nn_loaded, state_loaded = self.cm.load_training_checkpoint(fp)
        assert state_loaded.episode == 100
        assert state_loaded.epsilon == 0.5

    def test_auto_save_trigger(self):
        assert self.cm.should_auto_save(1000)
        assert self.cm.should_auto_save(2000)
        assert not self.cm.should_auto_save(0)
        assert not self.cm.should_auto_save(500)

    def test_cleanup_old(self):
        cm = CheckpointManager(
            checkpoint_dir=tempfile.mkdtemp(), max_checkpoints=3, verbose=False
        )
        for i in range(5):
            state = TrainingState(
                episode=i * 100,
                epsilon=0.5,
                best_win_rate_rule=0.6,
                best_win_rate_random=0.7,
                learning_rate=0.001,
                total_episodes=1000,
                phase=0,
                metrics_history=[],
                timestamp="2026-04-13 12:00:00",
            )
            cm.save_training_checkpoint(self.nn, state)
        cps = cm.list_checkpoints("training")
        assert len(cps["training"]) == 3
        shutil.rmtree(cm.checkpoint_dir, ignore_errors=True)

    def test_save_and_load_game(self):
        state = GameState(
            board=[["X", None, "O"], [None, "X", None], [None, None, None]],
            current_player="O",
            move_history=[],
            game_mode="pve",
            ai_difficulty="medium",
            is_game_over=False,
            winner=None,
            timestamp="2026-04-13 12:00:00",
        )
        fp = self.cm.save_game_checkpoint(state)
        assert Path(fp).exists()
        loaded = self.cm.load_game_checkpoint(fp)
        assert loaded.current_player == "O"

    def test_delete_checkpoint(self):
        state = TrainingState(
            episode=100,
            epsilon=0.5,
            best_win_rate_rule=0.6,
            best_win_rate_random=0.7,
            learning_rate=0.001,
            total_episodes=1000,
            phase=0,
            metrics_history=[],
            timestamp="2026-04-13 12:00:00",
        )
        fp = self.cm.save_training_checkpoint(self.nn, state)
        assert self.cm.delete_checkpoint(fp)
        assert not Path(fp).exists()
        assert not self.cm.delete_checkpoint("nonexistent.pkl")

    def test_clear_all(self):
        state = TrainingState(
            episode=100,
            epsilon=0.5,
            best_win_rate_rule=0.6,
            best_win_rate_random=0.7,
            learning_rate=0.001,
            total_episodes=1000,
            phase=0,
            metrics_history=[],
            timestamp="2026-04-13 12:00:00",
        )
        self.cm.save_training_checkpoint(self.nn, state)
        gs = GameState(
            board=[[None] * 3 for _ in range(3)],
            current_player="X",
            move_history=[],
            game_mode="pve",
            ai_difficulty="medium",
            is_game_over=False,
            winner=None,
            timestamp="2026-04-13 12:00:00",
        )
        self.cm.save_game_checkpoint(gs)
        self.cm.clear_all_checkpoints("all")
        cps = self.cm.list_checkpoints("all")
        assert len(cps["training"]) == 0
        assert len(cps["game"]) == 0


class TestFirstMoverConfig:
    def test_random_distribution(self):
        cfg = FirstMoverConfig(ai_first_ratio=0.5)
        x_count = sum(1 for _ in range(200) if cfg.get_first_player() == "X")
        assert 60 < x_count < 140

    def test_fixed_x(self):
        cfg = FirstMoverConfig(fixed_first_player="X")
        for i in range(10):
            assert cfg.get_first_player(i) == "X"

    def test_fixed_o(self):
        cfg = FirstMoverConfig(fixed_first_player="O")
        for i in range(10):
            assert cfg.get_first_player(i) == "O"

    def test_alternate(self):
        cfg = FirstMoverConfig(alternate_turns=True)
        for i in range(10):
            expected = "X" if i % 2 == 0 else "O"
            assert cfg.get_first_player(i) == expected

    def test_ratio_0(self):
        cfg = FirstMoverConfig(ai_first_ratio=0.0)
        for _ in range(10):
            assert cfg.get_first_player() == "O"

    def test_ratio_1(self):
        cfg = FirstMoverConfig(ai_first_ratio=1.0)
        for _ in range(10):
            assert cfg.get_first_player() == "X"


class TestTrainingState:
    def test_to_dict(self):
        state = TrainingState(
            episode=100,
            epsilon=0.5,
            best_win_rate_rule=0.6,
            best_win_rate_random=0.7,
            learning_rate=0.001,
            total_episodes=1000,
            phase=0,
            metrics_history=[],
            timestamp="2026-04-13 12:00:00",
        )
        d = state.to_dict()
        assert d["episode"] == 100
        assert d["version"] == "1.0"

    def test_from_dict(self):
        d = {
            "episode": 100,
            "epsilon": 0.5,
            "best_win_rate_rule": 0.6,
            "best_win_rate_random": 0.7,
            "learning_rate": 0.001,
            "total_episodes": 1000,
            "phase": 0,
            "metrics_history": [],
            "timestamp": "2026-04-13 12:00:00",
            "version": "1.0",
        }
        state = TrainingState.from_dict(d)
        assert state.episode == 100


class TestGameStateDataclass:
    def test_to_dict(self):
        state = GameState(
            board=[[None] * 3 for _ in range(3)],
            current_player="X",
            move_history=[],
            game_mode="pve",
            ai_difficulty="medium",
            is_game_over=False,
            winner=None,
            timestamp="2026-04-13 12:00:00",
        )
        d = state.to_dict()
        assert d["current_player"] == "X"

    def test_from_dict(self):
        d = {
            "board": [["X", None, None], [None, None, None], [None, None, None]],
            "current_player": "X",
            "move_history": [],
            "game_mode": "pve",
            "ai_difficulty": "medium",
            "is_game_over": False,
            "winner": None,
            "timestamp": "2026-04-13 12:00:00",
            "version": "1.0",
        }
        state = GameState.from_dict(d)
        assert state.current_player == "X"
