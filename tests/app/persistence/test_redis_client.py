import pytest
import json
import redis # Import redis to access its exceptions
from app.persistence.redis_client import RedisClient

@pytest.fixture
def mock_redis_client(mocker):
    """
    Fixture to mock the redis.Redis client and its methods.
    """
    mock_redis = mocker.patch('redis.Redis', autospec=True)
    mock_instance = mock_redis.return_value

    # Mock the ping to simulate a successful connection
    mock_instance.ping.return_value = True

    # Attach the mock instance to the fixture so we can inspect it
    mocker.mock_instance = mock_instance
    return mocker

def test_redis_client_init_success(mock_redis_client):
    """
    Test successful initialization of RedisClient.
    """
    client = RedisClient()
    assert client.is_connected()
    mock_redis_client.mock_instance.ping.assert_called_once()

def test_redis_client_init_failure(mocker):
    """
    Test connection failure during RedisClient initialization.
    """
    # Use the correct exception from the redis library
    mocker.patch('redis.Redis.ping', side_effect=redis.exceptions.ConnectionError("Connection failed"))
    client = RedisClient()
    assert not client.is_connected()

def test_set_json(mock_redis_client):
    """
    Test the set_json method.
    """
    client = RedisClient()
    test_data = {'a': 1, 'b': 'test'}
    client.set_json('mykey', test_data)

    # Assert that the client's 'set' method was called with the correct arguments
    mock_redis_client.mock_instance.set.assert_called_once_with('mykey', json.dumps(test_data))

def test_set_json_handles_exception(mock_redis_client):
    """Test that set_json returns False on exception."""
    mock_redis_client.mock_instance.set.side_effect = redis.exceptions.RedisError("Failed to set")
    client = RedisClient()
    assert client.set_json('mykey', {'a': 1}) is False

def test_get_json_found(mock_redis_client):
    """
    Test the get_json method when the key is found.
    """
    client = RedisClient()
    test_data = {'a': 1, 'b': 'test'}
    mock_redis_client.mock_instance.get.return_value = json.dumps(test_data)

    result = client.get_json('mykey')

    mock_redis_client.mock_instance.get.assert_called_once_with('mykey')
    assert result == test_data

def test_get_json_not_found(mock_redis_client):
    """
    Test the get_json method when the key is not found.
    """
    client = RedisClient()
    mock_redis_client.mock_instance.get.return_value = None

    result = client.get_json('mykey')

    mock_redis_client.mock_instance.get.assert_called_once_with('mykey')
    assert result is None

def test_delete(mock_redis_client):
    """Test the delete method."""
    client = RedisClient()
    client.delete('mykey')
    mock_redis_client.mock_instance.delete.assert_called_once_with('mykey')

def test_hset_json(mock_redis_client):
    """
    Test the hset_json method.
    """
    client = RedisClient()
    test_data = {'user': 'jules'}
    client.hset_json('myhash', 'user1', test_data)

    mock_redis_client.mock_instance.hset.assert_called_once_with('myhash', 'user1', json.dumps(test_data))

def test_hget_json_found(mock_redis_client):
    """
    Test the hget_json method when the field is found.
    """
    client = RedisClient()
    test_data = {'user': 'jules'}
    mock_redis_client.mock_instance.hget.return_value = json.dumps(test_data)

    result = client.hget_json('myhash', 'user1')

    mock_redis_client.mock_instance.hget.assert_called_once_with('myhash', 'user1')
    assert result == test_data

def test_hdel(mock_redis_client):
    """Test the hdel method."""
    client = RedisClient()
    client.hdel('myhash', 'user1')
    mock_redis_client.mock_instance.hdel.assert_called_once_with('myhash', 'user1')

def test_hgetall_json(mock_redis_client):
    """
    Test the hgetall_json method.
    """
    client = RedisClient()
    test_data = {
        'user1': json.dumps({'name': 'jules'}),
        'user2': json.dumps({'name': 'agent'})
    }
    expected_result = {
        'user1': {'name': 'jules'},
        'user2': {'name': 'agent'}
    }
    mock_redis_client.mock_instance.hgetall.return_value = test_data

    result = client.hgetall_json('myhash')

    mock_redis_client.mock_instance.hgetall.assert_called_once_with('myhash')
    assert result == expected_result
