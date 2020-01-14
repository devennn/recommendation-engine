import sys
from train import *

if(sys.argv[1] == 'new_user'):
    train_new_user()

elif(sys.argv[1] == 'existing_user'):
    train_existing_user()

else:
    print('Not found :: {}'.format(sys.argv[1]))
