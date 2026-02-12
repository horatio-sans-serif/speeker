#!/usr/bin/env python3
"""Unit tests for config.py functions."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from speeker.config import (
    get_config,
    save_config,
    is_semantic_search_enabled,
    get_embedding_model,
    get_embedding_cache_dir,
    get_llm_config,
)


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_dict(self, tmp_path):
        """Test that get_config returns a dictionary."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = get_config()
            assert isinstance(config, dict)

    def test_get_config_has_semantic_search_key(self, tmp_path):
        """Test config has semantic_search section."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = get_config()
            assert "semantic_search" in config

    def test_get_config_has_llm_key(self, tmp_path):
        """Test config has llm section."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = get_config()
            assert "llm" in config

    def test_get_config_missing_file_returns_defaults(self, tmp_path):
        """Test missing config file returns defaults."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = get_config()
            assert "semantic_search" in config

    def test_get_config_merges_with_defaults(self, tmp_path):
        """Test that loaded config is merged with defaults."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = get_config()
            assert "enabled" in config.get("semantic_search", {})
            assert "model" in config.get("semantic_search", {})


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config_creates_valid_json(self, tmp_path):
        """Test that save_config creates valid JSON."""
        test_config = {"test": "value"}

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config(test_config)

            config_file = tmp_path / "config" / "config.json"
            assert config_file.exists()

            with open(config_file) as f:
                loaded = json.load(f)
            assert loaded == test_config


class TestIsSemanticSearchEnabled:
    """Tests for is_semantic_search_enabled function."""

    def test_is_semantic_search_enabled_returns_bool(self, tmp_path):
        """Test returns boolean."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = is_semantic_search_enabled()
            assert isinstance(result, bool)

    @patch("speeker.config.get_config")
    def test_is_semantic_search_enabled_true(self, mock_get_config):
        """Test returns True when enabled."""
        mock_get_config.return_value = {"semantic_search": {"enabled": True}}
        assert is_semantic_search_enabled() is True

    @patch("speeker.config.get_config")
    def test_is_semantic_search_enabled_false(self, mock_get_config):
        """Test returns False when disabled."""
        mock_get_config.return_value = {"semantic_search": {"enabled": False}}
        assert is_semantic_search_enabled() is False

    @patch("speeker.config.get_config")
    def test_is_semantic_search_enabled_missing_key(self, mock_get_config):
        """Test returns False when key is missing."""
        mock_get_config.return_value = {}
        assert is_semantic_search_enabled() is False


class TestGetEmbeddingModel:
    """Tests for get_embedding_model function."""

    def test_get_embedding_model_returns_string(self, tmp_path):
        """Test returns string."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_embedding_model()
            assert isinstance(result, str)

    @patch("speeker.config.get_config")
    def test_get_embedding_model_returns_configured_value(self, mock_get_config):
        """Test returns configured model name."""
        mock_get_config.return_value = {"semantic_search": {"model": "custom-model"}}
        assert get_embedding_model() == "custom-model"

    @patch("speeker.config.get_config")
    def test_get_embedding_model_returns_default(self, mock_get_config):
        """Test returns default when not configured."""
        mock_get_config.return_value = {"semantic_search": {}}
        assert get_embedding_model() == "all-MiniLM-L6-v2"


class TestGetEmbeddingCacheDir:
    """Tests for get_embedding_cache_dir function."""

    @patch("speeker.config.get_config")
    def test_get_embedding_cache_dir_returns_none_by_default(self, mock_get_config):
        """Test returns None when not configured."""
        mock_get_config.return_value = {"semantic_search": {"cache_dir": None}}
        assert get_embedding_cache_dir() is None

    @patch("speeker.config.get_config")
    def test_get_embedding_cache_dir_returns_configured_path(self, mock_get_config):
        """Test returns configured path."""
        mock_get_config.return_value = {"semantic_search": {"cache_dir": "/tmp/models"}}
        assert get_embedding_cache_dir() == "/tmp/models"


class TestGetLlmConfig:
    """Tests for get_llm_config function."""

    def test_get_llm_config_returns_dict(self, tmp_path):
        """Test returns dictionary."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_llm_config()
            assert isinstance(result, dict)

    def test_get_llm_config_has_required_keys(self, tmp_path):
        """Test result has all required keys."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_llm_config()
            assert "backend" in result
            assert "endpoint" in result
            assert "api_key" in result
            assert "model" in result

    @patch("speeker.config.get_config")
    def test_get_llm_config_from_config_file(self, mock_get_config):
        """Test loads values from config file."""
        mock_get_config.return_value = {
            "llm": {
                "backend": "anthropic",
                "endpoint": "https://api.anthropic.com",
                "api_key": "test-key",
                "model": "claude-3",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm_config()
            assert result["backend"] == "anthropic"
            assert result["model"] == "claude-3"

    @patch("speeker.config.get_config")
    def test_get_llm_config_env_overrides_config(self, mock_get_config):
        """Test environment variables override config file."""
        mock_get_config.return_value = {
            "llm": {
                "backend": "anthropic",
                "model": "claude-3",
            }
        }
        env_vars = {
            "SPEEKER_LLM_BACKEND": "openai",
            "SPEEKER_LLM_MODEL": "gpt-4",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            result = get_llm_config()
            assert result["backend"] == "openai"
            assert result["model"] == "gpt-4"

    @patch("speeker.config.get_config")
    def test_get_llm_config_backend_lowercased(self, mock_get_config):
        """Test backend is lowercased."""
        mock_get_config.return_value = {"llm": {"backend": "ANTHROPIC"}}
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm_config()
            assert result["backend"] == "anthropic"

    @patch("speeker.config.get_config")
    def test_get_llm_config_missing_values_are_none(self, mock_get_config):
        """Test missing values default to None."""
        mock_get_config.return_value = {"llm": {}}
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm_config()
            assert result["backend"] is None
            assert result["endpoint"] is None
            assert result["api_key"] is None
            assert result["model"] is None
