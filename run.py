from config import Hyperparams
from agents import get_agent
from envs import get_env
from models import get_model
# main function to run the code
# on branch kevin
if __name__ == '__main__':
    args = Hyperparams().parse_args()
    model = get_model()
    env = get_env()
    agent = get_agent()