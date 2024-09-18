class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'BUSI':
            return 'data/BUSI/'

        elif dataset == 'BUDB':
            return 'data/BUDB/'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
