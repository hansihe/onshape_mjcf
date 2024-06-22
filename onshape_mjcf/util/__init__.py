import re
from colorama import Fore, Style

def formatName(name: str) -> str:
    match = re.fullmatch(r"^([\w_\. ]+)<[0-9]+>$", name)
    if match is not None:
        name = match.group(1)
    return name.strip().lower().replace(" ", "_")

def warn(string):
    print(Fore.YELLOW + "Warning: " + string + Style.RESET_ALL)

def error(string):
    print(Fore.RED + "Error: " + string + Style.RESET_ALL)
    raise Exception(string)