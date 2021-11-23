"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'impact_kb', 'impact_full_balanced', 'impact_full_imbalanced',
                    'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200', 'hdi_balanced', 'hdi_imbalanced',
                    'tobacco3482', 'rvl-cdip', 'wpi_demo'}
        assert(database in db_names)

        if database == 'cifar-10':
            return './cifar-10/'
        
        elif database == 'cifar-20':
            return './cifar-20/'

        elif database == 'stl-10':
            return './stl-10/'

        elif database == 'impact_kb':
            return './IMPACT_KB_240/'

        elif database == 'impact_full_balanced':
            return './IMPACT_Full_Balanced_240/'

        elif database == 'impact_full_imbalanced':
            return './IMPACT_Full_Imbalanced_240/'

        elif database == 'hdi_balanced':
            return './HDI_balanced/'

        elif database == 'hdi_imbalanced':
            return './HDI_imbalanced/'

        elif database == 'tobacco3482':
            return './tobacco3482/'

        elif database == 'rvl-cdip':
            return './rvl-cdip-240/'

        elif database == 'wpi_demo':
            return './wpi_data/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return './imagenet/'
        
        else:
            raise NotImplementedError
