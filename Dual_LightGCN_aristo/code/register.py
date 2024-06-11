import world
import dataloader
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'goodbooks-10k']:
    if world.config['dual']:
        dataset_good = dataloader.Loader(path="../data/"+world.dataset, dual=True, dual_case='good')
        dataset_bad = dataloader.Loader(path="../data/"+world.dataset, dual=True, dual_case='bad')
    else:
        dataset = dataloader.Loader(path="../data/"+world.dataset)


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')