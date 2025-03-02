import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

logger = logging.getLogger(__name__)


def extract_json_from_model_output(content: str) -> dict:
    """Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
    try:
        # If content is wrapped in code blocks, extract just the JSON part
        if '```' in content:
            # Find the JSON content between code blocks
            content = content.split('```')[1]
            # Remove language identifier if present (e.g., 'json\n')
            if '\n' in content:
                content = content.split('\n', 1)[1]
        # Parse the cleaned content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f'Failed to parse model output: {content} {str(e)}')
        raise ValueError('Could not parse response.')


# Define generic type variables for return type and parameters
R = TypeVar('R')
P = ParamSpec('P')


def time_execution_sync(additional_text: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:
	def decorator(func: Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = func(*args, **kwargs)
			execution_time = time.time() - start_time
			logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
			return result

		return wrapper

	return decorator


def time_execution_async(
	additional_text: str = '',
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
	def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
		@wraps(func)
		async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = await func(*args, **kwargs)
			execution_time = time.time() - start_time
			logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
			return result

		return wrapper

	return decorator


def singleton(cls):
	instance = [None]

	def wrapper(*args, **kwargs):
		if instance[0] is None:
			instance[0] = cls(*args, **kwargs)
		return instance[0]

	return wrapper
