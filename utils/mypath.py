"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'impact_kb', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return './cifar-10/'
        
        elif database == 'cifar-20':
            return './cifar-20/'

        elif database == 'stl-10':
            return './stl-10/'

        if database == 'impact_kb':
            return './IMPACT_KB_240/'

        if database == 'impact_full_balanced':
            return './IMPACT_Full_Balanced_240/'

        if database == 'impact_full_imbalanced':
            return './IMPACT_Full_Imbalanced_240/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return './imagenet/'
        
        else:
            raise NotImplementedError
