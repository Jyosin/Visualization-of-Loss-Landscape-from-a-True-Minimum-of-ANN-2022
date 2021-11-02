import ruamel.yaml as yaml
from colorama import Fore

ERROR_INFO = 'ERROR: '
NORMAL_INFO = 'INFO: '
WARNING_INFO = 'WARNING'

def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.scanner.ScannerError as err:
        print_error('yaml file format error!')
        print_error(err)
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)

def print_error(*content):
    '''Print error informaation to screen'''
    print(Fore.RED +  ERROR_INFO + ' '.join([str(c) for c in content]) + Fore.RESET)