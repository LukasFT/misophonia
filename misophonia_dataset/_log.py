import sys
import time

import eliot

PRINT_LOG_IS_SETUP = False


def setup_print_logging() -> None:
    global PRINT_LOG_IS_SETUP
    if PRINT_LOG_IS_SETUP:
        return

    def _printer(message: dict) -> None:
        # eg.:
        # {'timestamp': 1763563901.5133538, 'task_uuid': 'fd5608db-c4ba-4273-ac74-e2f115a94ee0', 'task_level': [1], 'message_type': 'This is a playground for testing code snippets.'}
        # Format pretty
        level = message.get("level", "").upper()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(message["timestamp"]))
        # print(f"[{time_str}] [{level}] {message['message_type']}")
        # add colors:
        level_color = {
            "DEBUG": "\033[94m",  # Blue
            "": "\033[0m",  # Default
            "INFO": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
        }.get(level, "")  # Default to no color
        reset_color = "\033[0m"
        print(
            f"{level_color}[{time_str}] [{level}]{reset_color} {message['message_type']}",
            flush=True,
            file=sys.stderr if message.get("to_stderr", False) else None,
        )

    eliot.add_destinations(_printer)

    eliot.log_message("Set up print logging", level="debug")

    PRINT_LOG_IS_SETUP = True
