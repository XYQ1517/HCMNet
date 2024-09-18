from dataloaders.datasets import BUSI, BUDB
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from mypath import Path


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def make_data_loader(args, **kwargs):
    if args.dataset == 'BUSI':
        train_loader_list = []
        val_loader_list = []
        test_loader_list = []
        for i in range(5):
            base_dir = Path.db_root_dir(args.dataset)
            open_fold = base_dir + 'fold_' + str(i) + '/'

            train_set = BUSI.Segmentation(args, open_fold, split='train')
            val_set = BUSI.Segmentation(args, open_fold, split='val')
            test_set = BUSI.Segmentation(args, open_fold, split='val')

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)
            test_loader_list.append(test_loader)

        return train_loader_list, val_loader_list, test_loader_list, num_class

    elif args.dataset == 'BUDB':
        train_loader_list = []
        val_loader_list = []
        test_loader_list = []
        for i in range(5):
            base_dir = Path.db_root_dir(args.dataset)
            open_fold = base_dir + 'fold_' + str(i) + '/'

            train_set = BUDB.Segmentation(args, open_fold, split='train')
            val_set = BUDB.Segmentation(args, open_fold, split='val')
            test_set = BUDB.Segmentation(args, open_fold, split='val')

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)
            test_loader_list.append(test_loader)

        return train_loader_list, val_loader_list, test_loader_list, num_class

    else:
        raise NotImplementedError

