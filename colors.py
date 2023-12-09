class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BACKGROUND_BLACK = '\033[40m'
    BACKGROUND_RED = '\033[41m'
    BACKGROUND_GREEN = '\033[42m'
    BACKGROUND_YELLOW = '\033[43m'

    
    @staticmethod
    def colorize(color, text, bold=False):
        if bold:
            return color + Color.BOLD + text + Color.RESET
        return color + text + Color.RESET
    def examples():
        print(Color.colorize(Color.BLACK, 'BLACK'))
        print(Color.colorize(Color.RED, 'RED'))
        print(Color.colorize(Color.GREEN, 'GREEN'))
        print(Color.colorize(Color.YELLOW, 'YELLOW'))
        print(Color.colorize(Color.BLUE, 'BLUE'))
        print(Color.colorize(Color.MAGENTA, 'MAGENTA'))
        print(Color.colorize(Color.CYAN, 'CYAN'))
        print(Color.colorize(Color.WHITE, 'WHITE'))
        print(Color.colorize(Color.BRIGHT_BLACK, 'BRIGHT_BLACK'))
        print(Color.colorize(Color.BRIGHT_RED, 'BRIGHT_RED'))
        print(Color.colorize(Color.BRIGHT_GREEN, 'BRIGHT_GREEN'))
        print(Color.colorize(Color.BRIGHT_YELLOW, 'BRIGHT_YELLOW'))
        print(Color.colorize(Color.BRIGHT_BLUE, 'BRIGHT_BLUE'))
        print(Color.colorize(Color.BRIGHT_MAGENTA, 'BRIGHT_MAGENTA'))
        print(Color.colorize(Color.BRIGHT_CYAN, 'BRIGHT_CYAN'))
        print(Color.colorize(Color.BRIGHT_WHITE, 'BRIGHT_WHITE'))
        print(Color.colorize(Color.BACKGROUND_BLACK, 'BACKGROUND_BLACK'))
        print(Color.colorize(Color.BACKGROUND_RED, 'BACKGROUND_RED'))
        print(Color.colorize(Color.BACKGROUND_GREEN, 'BACKGROUND_GREEN'))
        print(Color.colorize(Color.BACKGROUND_YELLOW, 'BACKGROUND_YELLOW'))