"""
Agent implementation.
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
import re
import uuid
from contextlib import AsyncExitStack
from enum import Enum
from time import time
from typing import (
	Any,
	Callable,
	Dict,
	Generic,
	List,
	Literal,
	Optional,
	Sequence,
	TypedDict,
	TypeVar,
	Union,
	get_args,
	cast,
)
import uuid

import langchain
from langchain.schema import (
	AIMessage,
	BaseMessage,
	ChatGeneration,
	ChatResult,
	HumanMessage,
	SystemMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError
from bs4 import BeautifulSoup

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.utils import convert_input_messages
from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.agent.views import AgentResults, AgentSettings, AgentState
from browser_use.browser.browser import Browser
from browser_use.utils import extract_json_from_model_output, time_execution_async

from .gif import generate_gif, get_gif_fp

logger = logging.getLogger('browser_use.agent')

Context = TypeVar('Context')


def _unpack_task_or_url_task_sync(task: Optional[str], url: Optional[str]) -> str:
	"""Unpacks a task or a url into a task (synchronous version)"""
	if not task and not url:
		raise ValueError('Either task or url must be specified')

	if task and url:
		if not 'http' in task:
			return task

	if url and not task:
		return f'Visit {url} and extract its main content'

	return task or ''

async def _unpack_task_or_url_task(task: Optional[str], url: Optional[str]) -> str:
	"""Unpacks a task or a url into a task (async version)"""
	# Same logic as the sync version, but async for compatibility
	return _unpack_task_or_url_task_sync(task, url)


class Agent(Generic[Context]):
	"""
	Agent that can perform tasks on behalf of a user.

	Args:
		task: The task to perform
		llm: The language model to use
		output_schema: The schema to use for collecting the result
		browser: The browser to use. Default to None, a new one will be created
		browser_options: Options for the browser
		system_prompt: The system prompt to use
		initial_actions: A list of actions to execute before running the main task
		task_callback: A callback that will be called when the task is done
		follow_up_task_callback: A callback that will be called with follow-up tasks
		conversation_saved_callback: A callback that will be called when the conversation is saved
		valid_urls: A list of regular expressions for URLs that can be visited
		maximum_web_actions: The maximum number of web actions to execute
		max_steps: The maximum number of steps to execute
		show_progress: Whether to write progress to stdout, useful for jupyter
		show_data_collection: Whether to show data collection
	"""

	THINK_TAGS = r'<think>(.*?)</think>'

	def __init__(
		self,
		task: Optional[str] = None,
		url: Optional[str] = None,
		llm: Optional[BaseChatModel] = None,
		output_schema: Optional[type[BaseModel]] = None,
		browser: Optional[Browser] = None,
		browser_options: Optional[Dict[str, Any]] = None,
		context: Optional[Context] = None,
		system_prompt: Optional[str] = None,
		user_agent: Optional[str] = None,
		initial_actions: Optional[list[dict[str, Any]]] = None,
		task_callback: Optional[Callable[[AgentResults], None]] = None,
		follow_up_task_callback: Optional[
			Callable[[list[dict[str, Any]]], None]
		] = None,
		conversation_saved_callback: Optional[Callable[[str], None]] = None,
		valid_urls: Optional[List[str]] = None,
		excluded_actions: Optional[List[str]] = None,
		maximum_web_actions: Optional[int] = None,
		max_steps: Optional[int] = None,
		save_conversation_path: Optional[str] = None,
		show_progress: bool = False,
		show_data_collection: bool = False,
		show_debug_completion: bool = False,
		tool_calling_method: Optional[Literal['raw', 'function', 'json', 'json_schema']] = None,
		page_extraction_llm: Optional[BaseChatModel] = None,
		planner_llm: Optional[BaseChatModel] = None,
		planner_interval: int = 1,  # Run planner every N steps
		cloudverse_endpoint: Optional[str] = None,  # Optional cloudverse API endpoint URL
		use_cloudverse: bool = False,  # Flag to use cloudverse instead of default LLM
		cloudverse_api_key: Optional[str] = None,  # API key for cloudverse authentication
		# Inject state
		injected_agent_state: Optional[AgentState] = None,
		#
		# Deprecated parameters
		additional_models: Optional[Dict[str, Any]] = None,
	):
		"""Initialize the agent"""
		del additional_models

		# Use the synchronous version to avoid asyncio.run() issues inside an event loop
		task = _unpack_task_or_url_task_sync(task, url)

		self.model_name = None if not llm else llm.model_name

		self.llm = llm
		self.output_schema = output_schema
		self.tool_calling_method = tool_calling_method or 'raw'

		self.task = task
		self.settings = AgentSettings(
			task=task,
			system_prompt=system_prompt,
			user_agent=user_agent,
			initial_actions=initial_actions,
			task_callback=task_callback,
			follow_up_task_callback=follow_up_task_callback,
			conversation_saved_callback=conversation_saved_callback,
			valid_urls=valid_urls,
			excluded_actions=excluded_actions,
			maximum_web_actions=maximum_web_actions,
			max_steps=max_steps,
			save_conversation_path=save_conversation_path,
			show_progress=show_progress,
			show_data_collection=show_data_collection,
			show_debug_completion=show_debug_completion,
			page_extraction_llm=page_extraction_llm,
			planner_llm=planner_llm,
			planner_interval=planner_interval,
			cloudverse_endpoint=cloudverse_endpoint,
			use_cloudverse=use_cloudverse,
			cloudverse_api_key=cloudverse_api_key,
		)

		# Initialize state
		if injected_agent_state is not None:
			self.state = injected_agent_state
		else:
			self.state = AgentState()
		self.state.browser_session_id = str(uuid.uuid4())

		# Make sure message_manager_state is properly initialized
		if not hasattr(self.state, "message_manager_state") or self.state.message_manager_state is None:
			self.state.message_manager_state = MessageManagerState()

		# Create a message manager
		system_content = system_prompt or "You are a helpful browser automation assistant."
		system_msg = SystemMessage(content=system_content)
		self._message_manager = MessageManager(task=self.task, system_message=system_msg, state=self.state.message_manager_state)

		# The user provided browser and ctx
		self.browser = browser
		self.ctx = context

		# If browser was provided, register it here and pass it to the message manager
		if browser:
			# Just register it
			try:
				self._message_manager.register_browser_session(self.state.browser_session_id, browser.page)
			except Exception as e:
				logger.warning(f"Failed to register browser session: {e}")

		# Set options
		self.browser_options = browser_options or {}

		self.browser_started = False
		self.default_exit_stack = AsyncExitStack()

	async def run(self, keep_browser_alive: bool = False) -> AgentResults:
		"""
		Run the agent.

		Args:
			keep_browser_alive: Whether to keep the browser alive after the run

		Returns:
			dict containing the following keys:
				- final_answer: The final answer from the model
				- follow_up_tasks: A list of follow-up tasks
				- self.state.history: The conversation history
		"""
		if self.settings.max_steps is not None and self.settings.max_steps <= 0:
			# Shortcut to just return the task
			answer = self.task
			follow_up_tasks = []
			message = AIMessage(content=answer)
			self.state.history.add_ai_message(message)
			return AgentResults(
				final_answer=answer,
				follow_up_tasks=follow_up_tasks,
				message_history=self.state.history,
			)

		try:
			# Only start browser if it's not provided
			if not self.browser:
				self.browser = await self._start_browser()
				# It was started by us, let's set the flag so we can close it
				self.browser_started = True

			# Register browser in the message manager
			try:
				if hasattr(self._message_manager, 'register_browser_session'):
					self._message_manager.register_browser_session(
						self.state.browser_session_id, self.browser.page
					)
			except Exception as e:
				logger.warning(f"Failed to register browser session: {e}")

			# Start the run
			system_message = await self._create_system_message()

			self._message_manager.add_system_message(system_message)

			# Execute initial actions
			if self.settings.initial_actions:
				for action in self.settings.initial_actions:
					try:
						await self._handle_action(action)
					except Exception as e:
						logger.warning(f'Error executing initial action: {e}')

			# Process the user input
			await self._process_user_input()

			# Handle empty browser inputs with a generic response
			final_answer = 'Task completed successfully.'
			follow_up_tasks = [
				{
					"title": "Continue exploring",
					"description": "Continue exploring this website."
				}
			]

			# The history of all messages
			message_history = self.state.history

			# Generate a GIF if requested
			if self.settings.generate_gif:
				file_path = get_gif_fp(str(self.settings.generate_gif))
				try:
					await generate_gif(self)
				except Exception as e:
					logger.exception(f'Error generating GIF: {e}')

			# The task callback
			if self.settings.task_callback:
				answer = self.state.history.get_answer()
				self.settings.task_callback(
					AgentResults(
						final_answer=answer or final_answer,
						follow_up_tasks=self.state.proposed_follow_up_tasks or [],
						message_history=message_history,
					)
				)

			# If we need to save the conversation
			if self.settings.save_conversation_path:
				self._save_conversation(self.settings.save_conversation_path)

			# Execute the follow-up task callback if any
			if self.settings.follow_up_task_callback and self.state.proposed_follow_up_tasks:
				self.settings.follow_up_task_callback(self.state.proposed_follow_up_tasks)

			return AgentResults(
				final_answer=self.state.history.get_answer() or final_answer,
				follow_up_tasks=self.state.proposed_follow_up_tasks or follow_up_tasks,
				message_history=message_history,
			)

		except Exception as e:
			# TODO do we need to handle errors here?
			raise e
		finally:
			if self.browser_started and not keep_browser_alive:
				await self.browser.close()

	async def _process_user_input(self) -> None:
		"""Process user input"""
		# Process the user input
		input_messages = await self._build_input_messages()
		tokens = self._message_manager.state.history.current_tokens

		try:
			model_output = await self.get_next_action(input_messages, self.settings.cloudverse_endpoint)

			self.state.n_steps += 1

			# Process the model output
			self.state.last_action = await self._process_model_output(model_output)

			# Update the thinking output
			if model_output.reasoning_process:
				self.state.history.add_thinking_message(model_output.reasoning_process)

			# If the model is ready to give the final answer, handle that
			if model_output.is_done:
				# The agent is done, let's add the final answer to the conversation
				self.state.history.add_ai_message(AIMessage(content=model_output.answer))

				# Mark done
				self.state.done = True

				if model_output.follow_up_tasks:
					self.state.proposed_follow_up_tasks = model_output.follow_up_tasks
			else:
				# If not done, we continue with the next action
				await self._handle_action(model_output.action)

				# Check if we should prompt the user again
				if not self.state.done and (
					self.settings.max_steps is None
					or self.state.n_steps < self.settings.max_steps
				):
					await self._process_user_input()
		except Exception as e:
			logger.error(f'Error processing user input: {e}')

			# If it's a retry, the AI will try to respond again
			await self._process_user_input()

	async def get_next_action(self, messages: List[BaseMessage], cloudverse_endpoint: Optional[str] = None) -> AgentOutput:
		"""Get the next action from the model"""
		# Track cloudverse usage mode
		if self.settings.use_cloudverse and self.settings.cloudverse_endpoint:
			try:
				return await self._get_next_action_from_cloudverse(messages, self.settings.cloudverse_endpoint, self.settings.cloudverse_api_key)
			except Exception as e:
				logger.error(f"Error using Cloudverse, falling back to standard LLM: {e}")
				
		elif not self.llm:
			raise ValueError("Either an LLM or Cloudverse endpoint must be provided")
			
		# Default to standard LLM processing
		result = await self.llm.agenerate([messages])
		
		generations: List[ChatGeneration] = result.generations[0]
		raw_result = generations[0].message
		
		return await self._extract_agent_output(raw_result)

	async def _get_next_action_from_cloudverse(self, input_messages: list[BaseMessage], cloudverse_endpoint: str, cloudverse_api_key: Optional[str] = None) -> "AgentOutput":
		"""Get the next action from Cloudverse API"""
		import aiohttp
		
		# Convert LangChain messages to standard format for Cloudverse API
		messages = []
		for message in input_messages:
			if isinstance(message, SystemMessage):
				messages.append({"role": "system", "content": message.content})
			elif isinstance(message, HumanMessage):
				messages.append({"role": "user", "content": message.content})
			elif isinstance(message, AIMessage):
				messages.append({"role": "assistant", "content": message.content})
		
		# Extract system instructions
		system_instructions = ""
		for message in input_messages:
			if isinstance(message, SystemMessage):
				system_instructions = message.content
				break
				
		# Prepare the API request payload
		payload = {
			"model": self.model_name,
			"messages": messages,
			"max_tokens": 2000,  # Default value, can be made configurable
			"temperature": 0,    # Default value, can be made configurable
			"top_p": 1,          # Default value, can be made configurable
			"system_instructions": system_instructions
		}
		
		try:
			# Setup headers with API key if provided
			headers = {}
			if cloudverse_api_key:
				headers["Authorization"] = f"Bearer {cloudverse_api_key}"
			
			async with aiohttp.ClientSession() as session:
				logger.info(f"Sending request to Cloudverse API at {cloudverse_endpoint}")
				async with session.post(cloudverse_endpoint, json=payload, headers=headers) as response:
					if response.status != 200:
						error_text = await response.text()
						logger.error(f"Cloudverse API error ({response.status}): {error_text}")
						raise ValueError(f"Cloudverse API returned error status {response.status}: {error_text}")
					
					response_data = await response.json()
					
					# Process the response based on its format
					if "choices" in response_data and len(response_data["choices"]) > 0:
						content = response_data["choices"][0].get("message", {}).get("content", "")
					else:
						content = response_data.get("content", "")
					
					# Extract the JSON part from the content if needed
					try:
						parsed_json = extract_json_from_model_output(content)
						agent_output = self.AgentOutput(**parsed_json)
						return agent_output
					except Exception as json_error:
						logger.error(f"Error parsing Cloudverse response: {json_error}, content: {content}")
						raise ValueError(f"Failed to parse Cloudverse response: {json_error}")
		
		except Exception as e:
			logger.error(f"Error communicating with Cloudverse API: {e}")
			raise

	async def _extract_agent_output(self, raw_response: BaseMessage) -> "AgentOutput":
		"""Extract the agent output from a raw response"""
		response_content = raw_response.content or ''
		
		# Try to extract JSON from the response
		try:
			# Extract reasoning from think tags
			reasoning_match = re.search(self.THINK_TAGS, response_content, re.DOTALL)
			reasoning_process = reasoning_match.group(1).strip() if reasoning_match else None
			
			# Remove thinking tag from content
			content_without_thinking = re.sub(self.THINK_TAGS, '', response_content, flags=re.DOTALL)
			
			# Parse the JSON and return as AgentOutput
			extracted_json = extract_json_from_model_output(content_without_thinking)
			
			# Create AgentOutput instance
			agent_output = self.AgentOutput(
				reasoning_process=reasoning_process,
				**extracted_json
			)
			
			return agent_output
		
		except Exception as e:
			logger.error(f"Error extracting agent output: {e} from response: {response_content}")
			raise ValueError(f"Failed to extract agent output: {e}")

	async def _process_model_output(self, model_output: "AgentOutput") -> Optional[Dict[str, Any]]:
		"""Process the model output"""
		# Validate the action
		if not model_output.is_done and not model_output.action:
			raise ValueError('Action must be provided if is_done is False')

		# Add model output to history
		self.state.history.add_model_output(model_output)

		# Validate the action
		if not model_output.is_done and model_output.action:
			if isinstance(model_output.action, list):
				# Multiple actions
				actions = model_output.action
				if len(actions) > self.settings.max_actions_per_step:
					logger.warning(f'More than {self.settings.max_actions_per_step} actions returned, using only first {self.settings.max_actions_per_step}')
					actions = actions[:self.settings.max_actions_per_step]

				# Process each action
				for action in actions:
					# Return the first action
					return action
			else:
				# Single action
				action = model_output.action
				return action

		return None

	async def _validate_output(self, result_json) -> bool:
		"""Validate the output of the model"""
		if not self.settings.validate_output:
			return True
		
		if hasattr(self, "validator") and self.validator:
			validator = self.validator
		else:
			from browser_use.agent.views import ValidationResult

			class ValidatorChain(BaseChatModel):
				"""Chain to validate the output of the model"""

				def __init__(self, llm: BaseChatModel):
					self.llm = llm

				async def _agenerate(self, messages, *args, **kwargs) -> ChatResult:
					"""Generate a validation result"""
					result = await self.llm.agenerate([messages], *args, **kwargs)
					generations: List[ChatGeneration] = result.generations[0]
					raw_result = generations[0].message
					extracted_json = extract_json_from_model_output(raw_result.content)
					# Since we know it's a validation result, we can just use this
					result_obj = ValidationResult(**extracted_json)
					return ChatResult(
						generations=[
							[
								ChatGeneration(
									message=AIMessage(
										content=json.dumps({'parsed': result_obj.model_dump()})
									)
								)
							]
						]
					)

				@property
				def _llm_type(self) -> str:
					"""Return the type of language model"""
					return 'validatorchain'

			validation_prompt = f"""
			As an validator assistant, your task is to check if an AI agent's action is valid and appropriate.
			I'll provide details about the action and you need to determine if it's valid.

			Please follow these evaluation guidelines:
			1. Check if the action has all the required fields
			2. Verify that all field values have valid types
			3. Ensure the action is appropriate for the current context
			4. Check if the action follows common web browsing patterns
			5. Don't validate overly complex actions that might abuse the browser or cause errors

			Your response must be in the following JSON format:
			{{
				"is_valid": true/false,
				"reason": "Brief explanation of why the action is valid or invalid"
			}}

			Here is the proposed action: {result_json}
			"""

			if self.settings.page_extraction_llm:
				plm = self.settings.page_extraction_llm
			elif self.llm:
				plm = self.llm
			else:
				return True  # Can't validate without a model

			msg = [HumanMessage(content=validation_prompt)]
			validator = ValidatorChain(plm)
			self.validator = validator
		
		try:
			# Check if we're using cloudverse
			if self.settings.use_cloudverse and self.settings.cloudverse_endpoint:
				import aiohttp
				
				# Setup headers with API key if provided
				headers = {}
				if self.settings.cloudverse_api_key:
					headers["Authorization"] = f"Bearer {self.settings.cloudverse_api_key}"
				
				# Convert to standard messages format for cloudverse API
				messages = []
				if isinstance(msg, list):
					for message in msg:
						if isinstance(message, SystemMessage):
							messages.append({"role": "system", "content": message.content})
						elif isinstance(message, HumanMessage):
							messages.append({"role": "user", "content": message.content})
						elif isinstance(message, AIMessage):
							messages.append({"role": "assistant", "content": message.content})
				
				import aiohttp
				import json
				import os
				
				# Ensure OpenAI API key is set for underlying libraries
				os.environ["OPENAI_API_KEY"] = self.settings.cloudverse_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
				
				# Prepare the API request payload
				payload = {
					"model": self.model_name or "gpt-4-turbo",
					"messages": messages,
					"max_tokens": 500,  # Default value for validation
					"temperature": 0,
					"top_p": 1,
					"system_instructions": "You are a helpful assistant that validates model output."
				}
				
				try:
					async with aiohttp.ClientSession() as session:
						async with session.post(self.settings.cloudverse_endpoint, json=payload, headers=headers) as response_http:
							if response_http.status != 200:
								error_text = await response_http.text()
								logger.error(f"Cloudverse API error ({response_http.status}): {error_text}")
								return True  # Default to valid on error
								
							response_data = await response_http.json()
							
							if "choices" in response_data and len(response_data["choices"]) > 0:
								content = response_data["choices"][0].get("message", {}).get("content", "")
							else:
								content = response_data.get("content", "")
								
							try:
								parsed_json = extract_json_from_model_output(content)
								response = {"parsed": ValidationResult(**parsed_json)}
							except Exception as e:
								logger.warning(f"Failed to parse cloudverse validation response: {e}")
								return True  # Default to valid on error
				except Exception as e:
					logger.error(f"Error validating with Cloudverse: {e}")
					return True  # Default to valid on error
			else:
				response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
			
			parsed: ValidationResult = response['parsed']
			is_valid = parsed.is_valid
			if not is_valid:
				logger.warning(f"Output validation failed: {parsed.reason}")
			
			return is_valid
		except Exception as e:
			logger.error(f"Error validating output: {e}")
			return True  # Default to valid on error

	async def _create_system_message(self) -> SystemMessage:
		"""Create a system message for the agent"""
		system_content = self.settings.system_prompt or "You are a helpful browser automation assistant."
		
		# Add user agent if specified
		if self.settings.user_agent:
			system_content += f"\nYou are using the following user agent: {self.settings.user_agent}"
			
		# Add excluded actions if specified
		if self.settings.excluded_actions:
			excluded_actions_str = ", ".join(self.settings.excluded_actions)
			system_content += f"\nThe following actions are not available: {excluded_actions_str}"
		
		# Create system message
		system_message = SystemMessage(content=system_content)
		return system_message

	async def _build_input_messages(self) -> list[BaseMessage]:
		return self._message_manager.get_input_messages()

	async def _handle_action(self, action: dict[str, Any]) -> None:
		"""Handle an action from the model"""
		if not action or not isinstance(action, dict):
			raise ValueError('Action must be a dictionary')
			
		# Get the action type
		action_type = action.get('type')
		
		if action_type == 'click':
			await self._handle_click(action)
		elif action_type == 'navigate':
			# Handle navigate action
			url = action.get('url')
			if url:
				if not self._is_valid_url(url):
					raise ValueError(
						f'URL {url} is not allowed, it does not match any of the valid URL patterns: {self.settings.valid_urls}'
					)

				# Navigate to the url
				await self.browser.go_to_page(url=url)
				await self._message_manager.update_page_visit_records()
				await self._message_manager.add_page_record()
				await self._message_manager.add_page_extraction_message()
			else:
				raise ValueError('URL not provided for navigation')
		elif action_type in ['extract_content', 'done', 'get_element_text', 'wait']:
			# These actions don't require browser interaction
			pass
		else:
			logger.warning(f'Unknown action type: {action_type}')
			
	async def _handle_click(self, action: dict[str, Any]) -> None:
		"""Handle click action"""
		# Extract the click parameters
		css_selector = action.get('css_selector')
		xpath = action.get('xpath')
		containing_text = action.get('containing_text')
		index = action.get('index')
		wait_for_navigation = action.get('wait_for_navigation', True)
		button_text = action.get('button_text')

		# Click the element with wait_for_navigation set to false initially
		if css_selector:
			await self.browser.click_element(
				css_selector=css_selector,
				wait_for_navigation=False,
				index=index,
			)
		elif xpath:
			await self.browser.click_element(
				xpath=xpath, wait_for_navigation=False, index=index
			)
		elif containing_text:
			await self.browser.click_element_containing_text(
				containing_text, wait_for_navigation=False, index=index
			)
		elif button_text:
			await self.browser.click_button_with_text(
				button_text, wait_for_navigation=False, index=index
			)
		else:
			raise ValueError(
				'css_selector, xpath, containing_text, or button_text must be provided for click action'
			)

		# Wait for navigation to complete if requested
		if wait_for_navigation:
			# We wait for navigation to complete
			await self.browser.wait_for_navigation()

			# Update the page visit records
			url = self.browser.page.url
			if not self._is_valid_url(url):
				raise ValueError(
					f'URL {url} is not allowed, it does not match any of the valid URL patterns: {self.settings.valid_urls}'
				)

			await self._message_manager.update_page_visit_records()
			await self._message_manager.add_page_record()
			await self._message_manager.add_page_extraction_message()

	def _is_valid_url(self, url: str) -> bool:
		"""Check if a URL is valid"""
		if not self.settings.valid_urls:
			return True

		for pattern in self.settings.valid_urls:
			if re.search(pattern, url):
				return True

		return False

	def _save_conversation(self, file_path: str) -> None:
		"""Save the conversation to a file"""
		import pickle

		# Save the conversation
		with open(file_path, 'wb') as f:
			pickle.dump(self.state.history, f)

		# Call the callback if any
		if self.settings.conversation_saved_callback:
			self.settings.conversation_saved_callback(file_path)

	async def _start_browser(self) -> Browser:
		"""Start a browser session"""
		from browser_use.browser.browser import Browser

		# Create browser instance without using async context manager
		browser = Browser(**self.browser_options)
		await browser.get_playwright_browser()  # Initialize the browser
		return browser

	class AgentOutput(BaseModel):
		"""Agent output"""

		reasoning_process: Optional[str] = Field(
			description="The reasoning process that led to the action or answer."
		)
		is_done: bool = Field(
			description="Whether the agent is done with the task. If True, an answer will be provided. If False, an action will be provided."
		)
		action: Optional[Dict[str, Any]] = Field(
			description="The next action to take. This could be web navigation, clicking, typing, etc., or instructing the model to provide a final answer. The format depends on the chosen action type."
		)
		answer: Optional[str] = Field(
			description="The final answer to the user's request. Only provided if is_done is True."
		)
		follow_up_tasks: Optional[list[dict[str, Any]]] = Field(
			description="A list of follow-up tasks that the user might want to do next. Each task is an object with a title and a description."
		)

	@property
	def message_manager(self) -> MessageManager:
		return self._message_manager