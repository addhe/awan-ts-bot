import pytest
import yaml
from unittest.mock import mock_open
from app import config

@pytest.fixture
def mock_env(mocker):
    """Fixture to mock environment variables."""
    env_vars = {
        'API_KEY_SPOT_BINANCE': 'test_api_key',
        'API_SECRET_SPOT_BINANCE': 'test_api_secret',
        'TELEGRAM_BOT_TOKEN': 'test_bot_token',
        'TELEGRAM_CHAT_ID': 'test_chat_id'
    }
    mocker.patch.dict('os.environ', env_vars)

@pytest.fixture
def mock_yaml_file(mocker):
    """Fixture to mock the config.yml file content."""
    yaml_content = """
    app:
      use_testnet: true
    redis:
      host: "mock_redis"
      port: 1234
    telegram:
      enabled: true
    """
    mocker.patch('builtins.open', mock_open(read_data=yaml_content))

def test_load_config_success(mock_env, mock_yaml_file):
    """
    Test that config loads correctly from both YAML and environment variables.
    """
    # We need to reload the config module to make it use the mocks
    import importlib
    importlib.reload(config)

    loaded_config = config.CONFIG

    # Assertions for YAML values
    assert loaded_config['app']['use_testnet'] is True
    assert loaded_config['redis']['host'] == 'mock_redis'

    # Assertions for environment variable values
    assert loaded_config['binance']['api_key'] == 'test_api_key'
    assert loaded_config['binance']['api_secret'] == 'test_api_secret'
    assert loaded_config['telegram']['bot_token'] == 'test_bot_token'
    assert loaded_config['telegram']['chat_id'] == 'test_chat_id'
    assert loaded_config['telegram']['enabled'] is True

def test_load_config_missing_telegram_keys(mocker, mock_yaml_file):
    """
    Test that Telegram is disabled if keys are missing.
    """
    # Simulate missing Telegram env vars
    mocker.patch.dict('os.environ', {
        'API_KEY_SPOT_BINANCE': 'test_api_key',
        'API_SECRET_SPOT_BINANCE': 'test_api_secret',
    })

    import importlib
    importlib.reload(config)

    loaded_config = config.CONFIG

    # Telegram should be disabled
    assert loaded_config['telegram']['enabled'] is False
    assert loaded_config['telegram']['bot_token'] is None
    assert loaded_config['telegram']['chat_id'] is None

def test_load_config_file_not_found(mocker):
    """
    Test that the application raises a FileNotFoundError if the config file is missing.
    """
    mocker.patch('builtins.open', side_effect=FileNotFoundError)

    import importlib
    with pytest.raises(FileNotFoundError):
        importlib.reload(config)
