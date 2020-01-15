import sys
from train import *
from recommend import *

def main():
        if(sys.argv[1] == 'new_user'):
            train_new_user()
            recommend_new()

        elif(sys.argv[1] == 'existing_user'):
            train_existing_user()
            recommend_existing()

        else:
            print('Not found :: {}'.format(sys.argv[1]))

if __name__ == '__main__':
    main()
