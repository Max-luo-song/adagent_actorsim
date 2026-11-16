import argparse
from action_validation import execute
p=argparse.ArgumentParser();p.add_argument('--folder',required=True)
if __name__=='__main__':execute(p.parse_args().folder)
