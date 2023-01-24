import logging
from subprocess import CalledProcessError, run
from textwrap import dedent


def run_shell(args, **kwargs):
    try:
        status = run(dedent(args).strip(), check=True, shell=True, **kwargs)
    except CalledProcessError as e:
        logging.error(e.stdout)
        logging.error(e.stderr)
        raise e
    return status
