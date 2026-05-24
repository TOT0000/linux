import datetime
import os
from typing import Union

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "debug", "verbose"}


def print_debug(*args, **kwargs):
    env_var = kwargs.pop("env_var", "TYPEFLY_DEBUG")
    if env_flag(env_var, default=False):
        print_t(*args, **kwargs)

def input_t(literal):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def split_args(arg_str: str) -> list[str]:
    print(f'split_args input: "{arg_str}"')
    args = []
    current_arg = ''
    parentheses_count = 0
    bracket_count = 0
    brace_count = 0
    in_quote = False
    quote_char = ''

    for char in arg_str:
        if in_quote:
            current_arg += char
            if char == quote_char:
                in_quote = False
            continue

        if char in ("'", '"'):
            in_quote = True
            quote_char = char
            current_arg += char
            continue

        if char == '(':
            parentheses_count += 1
        elif char == ')':
            parentheses_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        elif char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if (char == ',' and parentheses_count == 0 and bracket_count == 0 and brace_count == 0):
            if current_arg.strip():
                args.append(current_arg.strip())
            current_arg = ''
        else:
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args
