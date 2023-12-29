import os
import signal
import sys

import numpy as np

import colorama
from colorama import Fore, Back, Style

from utils import logging
from utils.exceptions import UserTerminationException

colorama.init()



class SimpleExecutor:
    def __init__(self, environment):
        """
        Initialize the SimpleExecutor with the current environment.

        :param environment: A dictionary representing the Python environment, typically `globals()`.
        """
        self.environment = environment


    def execute_plan(self, plan_code: str, additional_context: dict={}):
        """
        Execute the plan code within the Python environment held by the executor.

        :param plan_code: A string containing the Python code to be executed.
        """
        # Combine functions and temporary variables (masks)
        execution_context = self.environment.copy()
        execution_context.update(additional_context)
        exec(plan_code, execution_context)


class LineWiseExecutor:
    """
    A class for executing code with tracing and timeout capabilities.

    Attributes:
        environment (dict): The environment in which the code will be executed.
        timeout (int, optional): The maximum time in seconds allowed for execution.
        last_line (int): The last line of code executed.
        plan_code_lines (list): The lines of code to be executed.
        compiled_code: The compiled code object.

    Methods:

        execute_plan(plan_code, pre_execution_hook, post_execution_hook): Executes the plan code with optional hooks.

    Example:
        ```
        # Create an instance of the Executor with a mock environment
        class MockEnvironment(dict):
        def __init__(self, *args, **kwargs):
            super(MockEnvironment, self).__init__(*args, **kwargs)
            # Add the mock_function to the environment
            self['mock_function'] = self.mock_function

        def mock_function(self, message):
            print(f"Mock function says: {message}")

        executor = Executor(MockEnvironment(), timeout=5)

        # You can also do:
        def mock_function(message):
            print(f"Mock function says: {message}")
        executor = Executor(locals(), timeout=5)


        # A sample piece of plan code for testing
        plan_code = '''
        print("Starting execution...")
        for i in range(1, 4):
            print(f"Loop iteration: {i}")
            mock_function("Iteration " + str(i))
        print("Finished execution.")
        '''

        # Execute the plan using the Executor
        print("Test Execution Output:")
        executor.execute_plan(plan_code)
        ```

    """
    def __init__(self, environment, pause_every_line=False, enable_logging=True):
        """
        Initialize the Executor instance.

        This method sets up the execution environment for the Executor, 
        enabling it to run Python code within a specified environment. 
        It also allows for optional timeout functionality, which can 
        terminate the execution if it exceeds the specified duration.

        Parameters:
        - environment (dict): A dictionary representing the Python environment 
        in which the code will be executed. This should typically be the 
        `globals()` of the context in which the Executor is used, ensuring 
        that the executed code has access to the necessary functions, 
        variables, and imports.
        """
        self.environment = environment
        self.last_line = None
        self.plan_code_lines = None
        self.compiled_code = None  # Used to identify current python frame

        self.pause_every_line = pause_every_line
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.get_logger()

    def _handle_force_stop(self, signum, frame):
        """
        Signal handler for forced stop.
        """
        raise UserTerminationException("The user stopped the execution.")
    
    def _trace_function(self, frame, event, arg):
        if frame.f_code == self.compiled_code and event == "line":
            self.last_line = frame.f_lineno
            line = self.plan_code_lines[self.last_line - 1].strip()
            if self.pause_every_line:
                user_input = input(f"{Fore.YELLOW}(Executor) Next line to execute (press enter or 'y' to continue): {Fore.CYAN}{line}{Style.RESET_ALL}")
                if user_input not in ['']:
                    # Emit a signal to this process
                    # Get the current process ID
                    pid = os.getpid()
                    # Send a signal to the process
                    os.kill(pid, signal.SIGUSR1)
            if self.enable_logging:
                self.logger.log(name="Executor", log_type="info", message=f"Executing line {self.last_line}: {line}")
            print(f"{Fore.GREEN}(Executor) Executing line {self.last_line}: {Fore.CYAN}{line}{Style.RESET_ALL}")
        return self._trace_function

    def _execute_with_trace(self, code: str, env: dict):
        try:
            # Register a signal of the termination signal
            signal.signal(signal.SIGUSR1, self._handle_force_stop)
            self.plan_code_lines = code.split('\n')
            self.compiled_code = compile(code, "PlanCode", "exec")
            sys.settrace(self._trace_function)
            exec(code, env)
        except Exception as e:
            print(f"Error on line {self.last_line}: {str(e)}", file=sys.stderr)
            raise e
        finally:
            sys.settrace(None)


    def execute_plan(self, plan_code: str, additional_context: dict={}):
        # Combine functions and temporary variables
        execution_context = self.environment.copy()
        execution_context.update(additional_context)

        self._execute_with_trace(plan_code, execution_context)


class InspectExecutor:
    """
    InspectExecutor inspects the plan code by executing in a fake enviornment that only records the actions.
    """
    def __init__(self, environment):
        """
        Initialize the InspectExecutor with the current environment.

        :param environment: A dictionary representing the Python environment, typically `globals()`.
        """
        self.environment = environment


    def execute_plan(self, plan_code: str, additional_context: dict={}):
        """
        Execute the plan code within the Python environment held by the executor.

        :param plan_code: A string containing the Python code to be executed.
        """
        # Combine functions and temporary variables (masks)
        execution_context = self.environment.copy()
        execution_context.update(additional_context)
        exec(plan_code, execution_context)