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
from browser_use.agent.prompts import GIF_GENERATOR_SYSTEM_PROMPT
from browser_use.agent.views import AgentResults, AgentSettings, AgentState
from browser_use.browser.browser import Browser
from browser_use.utils import extract_json_from_model_output, time_execution_async

from .gif import generate_gif, get_gif_fp

logger = logging.getLogger('browser_use.agent')

Context = TypeVar('Context')


async def _unpack_task_or_url_task(task: Optional[str], url: Optional[str]) -> str:
	"""Unpacks a task or a url into a task"""
	# just to document the API

	if not task and not url:
		raise ValueError('Either task or url must be specified')

	if task and url:
		if not 'http' in task:
			return task

	if url and not task:
		return f'Visit {url} and extract its main content'

	return task or ''


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

		task = asyncio.run(_unpack_task_or_url_task(task, url))

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

		# Create a message manager
		self._message_manager = MessageManager(self.state)

		# The user provided browser and ctx
		self.browser = browser
		self.ctx = context

		# If browser was provided, register it here and pass it to the message manager
		if browser:
			# Just register it
			self._message_manager.register_browser_session(self.state.browser_session_id, browser.page)

		# Set options
		self.browser_options = browser_options or {}

		self.browser_started = False
		self.default_exit_stack = AsyncExitStack()

	async def run(self) -> AgentResults:
		"""
		Run the agent.

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
			self._message_manager.register_browser_session(
				self.state.browser_session_id, self.browser.page
			)

			# Start the run
			system_message = await self._create_system_message()

			# Add system message to the state
			self.state.history.add_system_message(system_message)

			# If no page is loaded, load the initial page
			if not getattr(self.browser.page, 'url', None) and self.browser_options.get(
				'initial_url'
			):
				initial_url = self.browser_options.get('initial_url')
				_ = await self.browser.go_to_page(url=initial_url)

			# Extract the current page content, if page has been loaded
			if self.browser.page.url:
				# Add page visit records
				if self.browser.page.url != 'about:blank':
					# Check if valid url, if valid_urls is set
					if not self._is_valid_url(self.browser.page.url):
						raise ValueError(
							f'URL {self.browser.page.url} is not allowed, it does not match any of the valid URL patterns: {self.settings.valid_urls}'
						)

					await self._message_manager.update_page_visit_records()
					await self._message_manager.add_page_record()

				# Extract data from the page
				await self._message_manager.add_page_extraction_message()

			# Step 1: If there are initial actions, execute them
			if self.settings.initial_actions:
				for action in self.settings.initial_actions:
					await self._handle_action(action)
					# If no more steps left, just stop
					if (
						self.settings.max_steps is not None
						and self.state.n_steps >= self.settings.max_steps
					):
						break

			# Step 2: Then start looping
			if self.state.n_steps == 0 or (
				self.settings.max_steps is not None
				and self.state.n_steps < self.settings.max_steps
			):
				# Add task to the history
				self.state.history.add_human_message(HumanMessage(content=self.task))
				await self._process_user_input()

			# Generate the results
			message_history = self.state.history
			# Get the last AI message

			if not message_history.ai_messages and self.settings.initial_actions:
				# If there are initial actions, but no AI messages, this can happen
				return AgentResults(
					final_answer='Initial actions executed, but no AI messages were generated.',
					follow_up_tasks=[],
					message_history=message_history,
				)

			last_message = (
				message_history.ai_messages[-1] if message_history.ai_messages else ''
			)
			final_answer = last_message.content if last_message else ''
			follow_up_tasks = (
				self.state.proposed_follow_up_tasks if self.state.proposed_follow_up_tasks else []
			)

			# Call the task callback
			if self.settings.task_callback:
				results = AgentResults(
					final_answer=final_answer,
					follow_up_tasks=follow_up_tasks,
					message_history=message_history,
				)
				self.settings.task_callback(results)

			# Call the follow-up task callback
			if follow_up_tasks and self.settings.follow_up_task_callback:
				self.settings.follow_up_task_callback(follow_up_tasks)

			# Save conversation
			if self.settings.save_conversation_path:
				self._save_conversation(self.settings.save_conversation_path)

			return AgentResults(
				final_answer=final_answer,
				follow_up_tasks=follow_up_tasks,
				message_history=message_history,
			)

		except Exception as e:
			# TODO do we need to handle errors here?
			raise e
		finally:
			if self.browser_started:
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
			raise e

	def _remove_think_tags(self, text: str) -> str:
		"""Remove <think> tags from the text"""
		if not text:
			return ""
		return re.sub(self.THINK_TAGS, '', text)

	def _convert_input_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Convert input messages to the correct format"""
		if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
			return convert_input_messages(input_messages, self.model_name)
		else:
			return input_messages

	@time_execution_async('--get_next_action (agent)')
	async def get_next_action(self, input_messages: list[BaseMessage], cloudverse_endpoint: Optional[str] = None) -> 'AgentOutput':
		"""Get next action from LLM based on current state
		
		Args:
			input_messages: List of messages to send to the LLM
			cloudverse_endpoint: Optional cloudverse API endpoint to use instead of the default LLM
		"""
		input_messages = self._convert_input_messages(input_messages)

		# Check if we should use cloudverse based on the settings
		if self.settings.use_cloudverse and self.settings.cloudverse_endpoint:
			cloudverse_endpoint = self.settings.cloudverse_endpoint
			cloudverse_api_key = self.settings.cloudverse_api_key
			
		if cloudverse_endpoint:
			# Use the cloudverse endpoint instead of the standard LLM
			import aiohttp
			import json
			import os
			
			# Ensure OpenAI API key is set for underlying libraries
			os.environ["OPENAI_API_KEY"] = cloudverse_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
			
			# Convert the input messages to the cloudverse API format
			messages = []
			for message in input_messages:
				if isinstance(message, SystemMessage):
					messages.append({"role": "system", "content": message.content})
				elif isinstance(message, HumanMessage):
					messages.append({"role": "user", "content": message.content})
				elif isinstance(message, AIMessage):
					messages.append({"role": "assistant", "content": message.content})
			
			# Extract system message for instructions
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
					async with session.post(cloudverse_endpoint, json=payload, headers=headers) as response:
						if response.status != 200:
							error_text = await response.text()
							logger.error(f"Cloudverse API error ({response.status}): {error_text}")
							raise ValueError(f"Cloudverse API returned error: {response.status}")
							
						response_data = await response.json()
						
						# Parse the model output based on the response format
						if "choices" in response_data and len(response_data["choices"]) > 0:
							content = response_data["choices"][0].get("message", {}).get("content", "")
						else:
							content = response_data.get("content", "")
							
						# Process the content to extract JSON
						try:
							parsed_json = extract_json_from_model_output(content)
							parsed = self.AgentOutput(**parsed_json)
						except (ValueError, ValidationError) as e:
							logger.warning(f"Failed to parse cloudverse output: {content} {str(e)}")
							raise ValueError("Could not parse cloudverse response.")
			except Exception as e:
				logger.error(f"Error calling cloudverse endpoint: {str(e)}")
				raise e
				
		elif self.tool_calling_method == 'raw':
			output = self.llm.invoke(input_messages)
			# TODO: currently invoke does not return reasoning_content, we should override invoke
			output.content = self._remove_think_tags(str(output.content))
			try:
				parsed_json = extract_json_from_model_output(output.content)
				parsed = self.AgentOutput(**parsed_json)
			except (ValueError, ValidationError) as e:
				logger.warning(f"Failed to parse model output: {output.content} {str(e)}")
				raise ValueError("Could not parse model output.")
			return parsed
		elif self.tool_calling_method == 'function':
			raise NotImplementedError(
				"Function calling doesn't work with langchain-core. Use json or json_schema"
			)
		elif self.tool_calling_method == 'json':
			result = self.llm.with_structured_output(
				self.AgentOutput,
			).invoke(input_messages)
			return result  # type: ignore
		elif self.tool_calling_method == 'json_schema':
			result = self.llm.with_structured_output(
				self.AgentOutput,
				include_raw=True,
			).invoke(input_messages)
			return result["parsed"]  # type: ignore
		else:
			raise ValueError(f"Unknown tool calling method: {self.tool_calling_method}")

		return parsed

	async def _process_model_output(self, output: 'AgentOutput') -> Optional[dict[str, Any]]:
		"""Process the model output"""
		# Check if the model wants to execute an action
		action_dict = output.action
		if action_dict:
			validate_action = bool(getattr(action_dict, 'validate', True))
			if validate_action:
				validated = await self._validate_output(output)
				if not validated:
					logger.warning(
						f"Model output validation failed. Retrying. Output: {output.action}"
					)
					return None
			
			return action_dict
		return None

	async def _handle_action(self, action: dict[str, Any]) -> None:
		"""Handle action such as navigation, clicking, etc."""
		try:
			# Skip action if excluded
			action_type = action.get('type')
			if action_type in self.settings.excluded_actions:
				logger.warning(f"Skipping excluded action: {action_type}")
				return

			# Update count of web actions, this is done before calling is_valid_url to ensure that
			# the count is accurate for all actions even failed ones and NavigateBack and NavigateForward,
			# which could return to invalid URLs

			if action_type in [
				'click',
				'navigate',
				'submit',
				'type',
				'navigate_back',
				'navigate_forward',
				'scroll',
				'save_conversation',
				'extract_text',
				'extract_dom',
				'extract_links',
				'show_data_collection',
				'answer',
				'follow_up_tasks',
				'load_cookies',
				'select',
				'generate_gif',
			]:
				self.state.web_actions += 1

				# If maximum number of web actions is reached, we stop
				if (
					self.settings.maximum_web_actions is not None
					and self.state.web_actions > self.settings.maximum_web_actions
				):
					logger.warning(
						f"Maximum number of web actions reached: {self.settings.maximum_web_actions}"
					)
					self.state.done = True
					return

			# Handle the different action types
			if action.get('type') == 'click':
				await self._handle_click(action)
			elif action.get('type') == 'navigate':
				url = action.get('url')
				if url:
					# Check if valid url
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
			elif action.get('type') == 'navigate_back':
				# Navigate back
				await self.browser.go_back()

				# Add to history
				await self._message_manager.add_page_record()
				await self._message_manager.add_page_extraction_message()
			elif action.get('type') == 'navigate_forward':
				# Navigate forward
				await self.browser.go_forward()

				# Add to history
				await self._message_manager.add_page_record()
				await self._message_manager.add_page_extraction_message()
			elif action.get('type') == 'submit':
				# Submit a form
				css_selector = action.get('css_selector')
				xpath = action.get('xpath')
				if css_selector:
					await self.browser.submit_form(css_selector=css_selector)
				elif xpath:
					await self.browser.submit_form(xpath=xpath)
				else:
					raise ValueError(
						'css_selector or xpath not provided for submit'
					)

				# Add to history
				await self._message_manager.add_page_record()
				await self._message_manager.add_page_extraction_message()
			elif action.get('type') == 'type':
				# Type text
				css_selector = action.get('css_selector')
				xpath = action.get('xpath')
				text = action.get('text')
				if not text:
					raise ValueError('Text not provided for typing')

				if css_selector:
					await self.browser.type_text(
						css_selector=css_selector, text=text
					)
				elif xpath:
					await self.browser.type_text(xpath=xpath, text=text)
				else:
					raise ValueError(
						'css_selector or xpath not provided for typing'
					)
			elif action.get('type') == 'select':
				# Select option
				css_selector = action.get('css_selector')
				xpath = action.get('xpath')
				value = action.get('value')
				label = action.get('label')
				index = action.get('index')

				if css_selector:
					await self.browser.select_option(
						css_selector=css_selector,
						value=value,
						label=label,
						index=index,
					)
				elif xpath:
					await self.browser.select_option(
						xpath=xpath, value=value, label=label, index=index
					)
				else:
					raise ValueError(
						'css_selector or xpath not provided for select'
					)
			elif action.get('type') == 'scroll':
				# Scroll the page
				direction = action.get('direction', 'down')
				amount = action.get('amount', '500px')
				css_selector = action.get('css_selector')
				if direction == 'down':
					if css_selector:
						await self.browser.scroll_down(
							css_selector=css_selector, amount=amount
						)
					else:
						await self.browser.scroll_down(amount=amount)
				elif direction == 'up':
					if css_selector:
						await self.browser.scroll_up(
							css_selector=css_selector, amount=amount
						)
					else:
						await self.browser.scroll_up(amount=amount)
				else:
					raise ValueError(
						f'Invalid scroll direction: {direction}. Must be "up" or "down"'
					)

				# If we're done scrolling, extract the page again
				if action.get('done_scrolling', False):
					await self._message_manager.add_page_extraction_message()
			elif action.get('type') == 'extract_text':
				css_selector = action.get('css_selector')
				xpath = action.get('xpath')
				if css_selector:
					extracted_text = await self.browser.extract_text(
						css_selector=css_selector
					)
				elif xpath:
					extracted_text = await self.browser.extract_text(xpath=xpath)
				else:
					raise ValueError(
						'css_selector or xpath not provided for extract_text'
					)

				# Log the extracted text
				extracted_type = "css_selector" if css_selector else "xpath"
				extracted_value = css_selector if css_selector else xpath
				message = f"""
Extracted text using {extracted_type} "{extracted_value}":
{extracted_text}
				"""
				self.state.history.add_execution_info_message(message)
			elif action.get('type') == 'extract_dom':
				extracted_dom = await self.browser.extract_dom()

				# Truncate the extracted dom
				if len(extracted_dom) > 1000:
					extracted_dom = extracted_dom[:1000] + '... (truncated)'

				# Log the extracted dom
				message = f"""
Extracted DOM:
{extracted_dom}
				"""
				self.state.history.add_execution_info_message(message)
			elif action.get('type') == 'extract_links':
				css_selector = action.get('css_selector')
				xpath = action.get('xpath')
				if css_selector:
					extracted_links = await self.browser.extract_links(
						css_selector=css_selector
					)
				elif xpath:
					extracted_links = await self.browser.extract_links(xpath=xpath)
				else:
					extracted_links = await self.browser.extract_links()

				# Format the extracted links
				formatted_links = '\n'.join(
					[f"- {link['text']}: {link['href']}" for link in extracted_links]
				)

				# Log the extracted links
				message = f"""
Extracted links:
{formatted_links}
				"""
				self.state.history.add_execution_info_message(message)
			elif action.get('type') == 'save_conversation':
				file_path = action.get('file_path')
				if file_path:
					self._save_conversation(file_path)
				else:
					raise ValueError('File path not provided for save_conversation')
			elif action.get('type') == 'load_cookies':
				cookies_file = action.get('cookies_file')
				if cookies_file:
					await self.browser.load_cookies(cookies_file)
				else:
					raise ValueError('Cookies file not provided for load_cookies')
			elif action.get('type') == 'show_data_collection':
				show = action.get('show', True)
				self.settings.show_data_collection = show
			elif action.get('type') == 'generate_gif':
				url = self.browser.page.url
				if 'youtube.com' in url or 'youtube.com/watch' in url:
					logger.info('Generating GIF for YouTube video')
					try:
						fp = get_gif_fp()
						await generate_gif(self.browser.page, fp, 500, 50)

						# Add image to state
						message = GIF_GENERATOR_SYSTEM_PROMPT.format(path=fp)
						self.state.history.add_execution_info_message(message)
					except Exception as e:
						logger.error(f'Error generating GIF: {e}')
				else:
					logger.warning('GIF generation is only supported for YouTube videos')
			elif action.get('type') == 'answer':
				# Add the final answer
				answer = action.get('answer')
				if answer:
					self.state.history.add_ai_message(AIMessage(content=answer))
					self.state.done = True
			elif action.get('type') == 'follow_up_tasks':
				follow_up_tasks = action.get('tasks')
				if follow_up_tasks:
					self.state.proposed_follow_up_tasks = follow_up_tasks
			else:
				raise ValueError(f'Unknown action type: {action.get("type")}')
		except Exception as e:
			logger.error(f'Error handling action: {e}')
			# Add to history
			message = f"""
Error handling action {action.get('type')}: {str(e)}
				"""
			self.state.history.add_execution_info_message(message)
			raise Exception(f"Error handling action: {e}")

	async def _validate_output(self, output: 'AgentOutput') -> bool:
		# TODO: Implement validation
		# TODO: If browser is not provided, we can't validate anything
		if not self.browser or not getattr(self.browser, 'page', None):
			# if no browser session, we can't validate the output
			return True

		class ValidationResult(BaseModel):
			"""
			Validation results.
			"""

			is_valid: bool
			reason: str

		validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
		
		# If cloudverse is enabled, use it for validation too
		if self.settings.use_cloudverse and self.settings.cloudverse_endpoint:
			# Setup headers with API key if provided
			headers = {}
			if self.settings.cloudverse_api_key:
				headers["Authorization"] = f"Bearer {self.settings.cloudverse_api_key}"
			
			# Convert to standard messages format for cloudverse API
			messages = []
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
				"model": self.model_name,
				"messages": messages,
				"max_tokens": 500,  # Default value for validation
				"temperature": 0,
				"top_p": 1,
				"system_instructions": system_msg
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

	async def _create_system_message(self) -> SystemMessage:
		# Add default system prompt
		from browser_use.agent.system_prompt import get_system_prompt

		# Load system prompt
		sp = await get_system_prompt(
			system_prompt=self.settings.system_prompt,
			user_agent=self.settings.user_agent,
			excluded_actions=self.settings.excluded_actions,
		)
		# Create system message
		system_message = SystemMessage(content=sp)
		return system_message

	async def _build_input_messages(self) -> list[BaseMessage]:
		return self._message_manager.get_input_messages()

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

		browser = await self.default_exit_stack.enter_async_context(
			Browser(**self.browser_options)
		)
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