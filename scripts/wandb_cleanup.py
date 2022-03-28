import logging
import contextlib
try:
    from http.client import HTTPConnection # py3
except ImportError:
    from httplib import HTTPConnection # py2

def debug_requests_on():
    '''Switches on logging of the requests module.'''
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def debug_requests_off():
    '''Switches off logging of the requests module, might be some side-effects'''
    HTTPConnection.debuglevel = 0

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = False

@contextlib.contextmanager
def debug_requests():
    '''Use with 'with'!'''
    debug_requests_on()
    yield
    debug_requests_off()

KEY = "166c81279a3c794e24a23332a7268d23800a7991"
PROJECT = "mosaic-gpt2"
ENTITY = "stanford-mercury"
# When using artifact api methods that don't have an entity or project
#  argument, you must provide that information when instantiating the wandb.Api
import wandb

wandb.login(key=KEY)

api = wandb.Api(overrides={"project": PROJECT, "entity": ENTITY})

for run in api.runs():
    files = sorted([f for f in run.logged_artifacts()], key=lambda f: f.updated_at)
    print("Total files:", len(files))
    print("Last file:", files[-1].name)
    print("Last file date:", files[-1].updated_at)
    for f in files[:-1]:
        # with debug_requests():
            if ".tar" in f.name:
                try:
                    # also tried just f.delete()
                    a = api.artifact(f"{PROJECT}/{f.name}")
                    print("Deleting {}".format(f.name))
                    a.delete()
                    print("Deleted {}".format(f.name))
                except Exception as e:
                    print("Failed to delete {}: {}".format(f.name, e))

