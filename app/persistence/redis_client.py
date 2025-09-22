import redis
import json
import logging
from app.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedisClient:
    def __init__(self):
        """
        Initializes the Redis client using settings from the global CONFIG.
        """
        redis_config = CONFIG.get('redis', {})
        host = redis_config.get('host', 'localhost')
        port = redis_config.get('port', 6379)
        db = redis_config.get('db', 0)

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                charset="utf-8",
                decode_responses=True # Decode responses to utf-8
            )
            # Ping the server to check the connection
            self.client.ping()
            logging.info(f"Successfully connected to Redis at {host}:{port}")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"Could not connect to Redis at {host}:{port}. Error: {e}")
            logging.error("Please ensure the Redis container is running and accessible.")
            self.client = None

    def is_connected(self):
        """
        Returns True if the client is connected to Redis, False otherwise.
        """
        return self.client is not None

    def set_json(self, key, data):
        """
        Serializes a Python dictionary to a JSON string and stores it in Redis.
        """
        if not self.is_connected():
            return None
        try:
            self.client.set(key, json.dumps(data))
            return True
        except Exception as e:
            logging.error(f"Error setting JSON data in Redis for key '{key}': {e}")
            return False

    def get_json(self, key):
        """
        Retrieves a JSON string from Redis and deserializes it to a Python dictionary.
        """
        if not self.is_connected():
            return None
        try:
            json_data = self.client.get(key)
            if json_data:
                return json.loads(json_data)
            return None
        except Exception as e:
            logging.error(f"Error getting JSON data from Redis for key '{key}': {e}")
            return None

    def delete(self, key):
        """
        Deletes a key from Redis.
        """
        if not self.is_connected():
            return None
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logging.error(f"Error deleting key '{key}' from Redis: {e}")
            return False

    def hset_json(self, hash_key, field, data):
        """
        Sets a field in a Redis hash with a JSON-serialized dictionary.
        """
        if not self.is_connected():
            return None
        try:
            self.client.hset(hash_key, field, json.dumps(data))
            return True
        except Exception as e:
            logging.error(f"Error setting hash field in Redis for key '{hash_key}': {e}")
            return False

    def hget_json(self, hash_key, field):
        """
        Gets a JSON-serialized dictionary from a field in a Redis hash.
        """
        if not self.is_connected():
            return None
        try:
            json_data = self.client.hget(hash_key, field)
            if json_data:
                return json.loads(json_data)
            return None
        except Exception as e:
            logging.error(f"Error getting hash field from Redis for key '{hash_key}': {e}")
            return None

    def hgetall_json(self, hash_key):
        """
        Gets all fields and values from a Redis hash and deserializes the JSON values.
        """
        if not self.is_connected():
            return None
        try:
            hash_data = self.client.hgetall(hash_key)
            return {field: json.loads(value) for field, value in hash_data.items()}
        except Exception as e:
            logging.error(f"Error getting all hash fields from Redis for key '{hash_key}': {e}")
            return None

    def hdel(self, hash_key, field):
        """
        Deletes a field from a Redis hash.
        """
        if not self.is_connected():
            return None
        try:
            self.client.hdel(hash_key, field)
            return True
        except Exception as e:
            logging.error(f"Error deleting hash field from Redis for key '{hash_key}': {e}")
            return False

# Create a singleton instance of the client to be used across the application
redis_client = RedisClient()
