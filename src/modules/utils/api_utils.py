import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

T = TypeVar('T')

def safe_api_call(
    func: Callable[..., T],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True
) -> T:
    """
    Safely execute an API call with retries and error handling.
    
    This decorator/wrapper provides robust error handling for API calls, including:
    - Automatic retries with exponential backoff
    - Rate limit handling
    - Network error recovery
    - Comprehensive logging
    
    Args:
        func: The API function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        exponential_backoff: Whether to use exponential backoff for retries (default: True)
        
    Returns:
        The result of the API call if successful
        
    Raises:
        Exception: Re-raises the last exception after all retries are exhausted
        
    Example:
        >>> balance = safe_api_call(exchange.fetch_balance)
        >>> # With custom retry parameters
        >>> ticker = safe_api_call(
        ...     exchange.fetch_ticker,
        ...     max_retries=5,
        ...     retry_delay=2.0
        ... )
    """
    if not callable(func):
        raise TypeError("func must be callable")
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate delay with exponential backoff if enabled
                    delay = (
                        retry_delay * (2 ** (attempt - 1))
                        if exponential_backoff
                        else retry_delay
                    )
                    logging.info(
                        f"Retry attempt {attempt}/{max_retries} "
                        f"for {func.__name__}, waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                logging.debug(
                    f"API call {func.__name__} successful "
                    f"(took {elapsed:.3f}s)"
                )
                
                return cast(T, result)
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Handle different types of errors
                if 'rate limit' in error_msg:
                    logging.warning(
                        f"Rate limit hit in {func.__name__}: {str(e)}"
                    )
                    # Always use exponential backoff for rate limits
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                    
                elif any(msg in error_msg for msg in [
                    'network',
                    'timeout',
                    'connection',
                    'socket'
                ]):
                    logging.warning(
                        f"Network error in {func.__name__}: {str(e)}"
                    )
                    continue
                    
                elif any(msg in error_msg for msg in [
                    'insufficient',
                    'balance',
                    'funds',
                    'minimum',
                    'maximum'
                ]):
                    # Don't retry balance/validation errors
                    logging.error(
                        f"Balance/validation error in {func.__name__}: {str(e)}"
                    )
                    raise
                    
                else:
                    logging.error(
                        f"Error in {func.__name__} (attempt {attempt + 1}): {str(e)}"
                    )
                    if attempt == max_retries:
                        break
                    continue
        
        # If we get here, all retries were exhausted
        if last_exception:
            logging.error(
                f"All {max_retries} retries exhausted for {func.__name__}"
            )
            raise last_exception
            
        # This should never happen since we either return or raise above
        raise RuntimeError(
            f"Unexpected error: all retries exhausted for {func.__name__}"
        )
    
    return wrapper
