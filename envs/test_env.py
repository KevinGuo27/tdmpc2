import os
import time
import unittest
from unittest.mock import patch
import gymnasium as gym
from __init__ import get_env  # Assume your function is in 'your_module.py'

class TestEnvironmentSetup(unittest.TestCase):
    def test_env_creation(self):
        env_id = 'Asterix-MinAtar'  # Example Atari game, change according to your use case
        env_creator = get_env(env_id)
        env = env_creator()  # This will call the thunk and create the environment

        # Check if the environment is created
        self.assertIsNotNone(env)

        # Check if the directory for videos is created
        run_name = f"{env_id}-{int(time.time())}"
        expected_path = f"videos/{run_name}"
        self.assertTrue(os.path.exists(expected_path))

        # Check that the environment has the expected wrappers
        # This is a simplistic check; you might need to verify the specific properties of each wrapper
        self.assertIsInstance(env, gym.wrappers.RecordVideo)
        self.assertIsInstance(env, gym.wrappers.RecordEpisodeStatistics)
        self.assertIsInstance(env, gym.wrappers.AutoResetWrapper)
        self.assertIsInstance(env, gym.wrappers.ClipAction)
        self.assertIsInstance(env, gym.wrappers.ResizeObservation)
        self.assertIsInstance(env, gym.wrappers.NormalizeObservation)
        self.assertIsInstance(env, gym.wrappers.GrayScaleObservation)
        self.assertIsInstance(env, gym.wrappers.FrameStack)

if __name__ == '__main__':
    unittest.main()