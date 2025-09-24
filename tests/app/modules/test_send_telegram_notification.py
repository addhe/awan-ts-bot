import pytest
from unittest.mock import patch
from app.modules.send_telegram_notification import send_telegram_notification

@patch('app.modules.send_telegram_notification.requests.post')
@patch.dict('os.environ', {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': 'test_id'})
def test_send_telegram_notification_success(mock_post):
    """
    Test that a successful notification calls requests.post.
    """
    message = "Hello, World!"
    send_telegram_notification(message)
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args.kwargs['json']['text'] == message

@patch('app.modules.send_telegram_notification.requests.post')
@patch.dict('os.environ', {}, clear=True)
def test_send_telegram_notification_no_creds(mock_post):
    """
    Test that no notification is sent if credentials are not set.
    """
    send_telegram_notification("This should not be sent")
    mock_post.assert_not_called()

@patch('app.modules.send_telegram_notification.requests.post', side_effect=Exception("API Error"))
@patch.dict('os.environ', {'TELEGRAM_BOT_TOKEN': 'test_token', 'TELEGRAM_CHAT_ID': 'test_id'})
def test_send_telegram_notification_handles_exception(mock_post):
    """
    Test that the function handles exceptions from requests.post gracefully.
    """
    # This test just ensures the function doesn't crash
    send_telegram_notification("Test message")
    mock_post.assert_called_once()
