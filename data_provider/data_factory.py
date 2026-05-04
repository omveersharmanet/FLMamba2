import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
    Dataset_Solar, Dataset_PEMS, Dataset_Pred
)

DATASET_TABLE = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}

ALL_DATASETS = [
    # Custom datasets

    ('electricity', './dataset/electricity/', 'electricity.csv', 'custom', 321,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    # ETT datasets
    ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 'ETTh1', 7,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 'ETTh2', 7,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 'ETTm1', 7,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 'ETTm2', 7,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    ('exchange', './dataset/exchange_rate/', 'exchange_rate.csv', 'custom', 8,[48, 96, 192 ,336 ,720 ,1024],[48 ,96 ,192 ,336 ,720 ,1024], [48, 48, 48, 48, 48, 48]),    
    # PEMS datasets (DIFFERENT pred lengths ✅)
    #('PEMS03', './dataset/PEMS/', 'PEMS03.npz', 'PEMS', 358,[48, 96, 192, 336, 720, 1024],[6, 12, 24, 48, 96, 192, 336, 720, 1024], [6,6,6,6,6,6,6,6,6]),
    #('PEMS04', './dataset/PEMS/', 'PEMS04.npz', 'PEMS', 307,[48, 96, 192, 336, 720, 1024],[6, 12, 24, 48, 96, 192, 336, 720, 1024], [6,6,6,6,6,6,6,6,6]),
   # ('PEMS07', './dataset/PEMS/', 'PEMS07.npz', 'PEMS', 883,[48, 96, 192, 336, 720, 1024],[6, 12, 24, 48, 96, 192, 336, 720, 1024], [6,6,6,6,6,6,6,6,6]),
    #('PEMS08', './dataset/PEMS/', 'PEMS08.npz', 'PEMS', 170,[48 ,96 ,192 ,336 ,720 ,1024],[6 ,12 ,24 ,48 ,96 ,192 ,336 ,720 ,1024], [6 ,6 ,6 ,6 ,6 ,6 ,6 ,6 ,6]),
    # Solar
    ('Solar', './dataset/Solar/', 'solar_AL.txt', 'Solar', 137,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]),
    ('traffic', './dataset/traffic/', 'traffic.csv', 'custom', 862,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48  ]),
    ('weather', './dataset/weather/', 'weather.csv', 'custom', 21,[48, 96, 192, 336, 720, 1024],[48, 96, 192, 336, 720, 1024],[48, 48, 48, 48, 48, 48]   ),     
]


def build_dataset(args, flag, dataset_cfg):
    _, root, path, dtype,enc,input_seq,prediction_seq,label_sequence = dataset_cfg
    DataCls = DATASET_TABLE[dtype]

    # Use Dataset_Pred for prediction
    #if flag == 'pred':
        #DataCls = Dataset_Pred

    return DataCls(
        root_path=root,
        data_path=path,
        flag=flag,
        size=[input_seq[args.seq_len_case],label_sequence[args.label_len_case], prediction_seq[args.pred_len_case]],
        features=args.features,
        target=args.target,
        timeenc=0 if args.embed != 'timeF' else 1,
        freq=args.freq,
    )

def data_provider(args, flag):
    """
    Returns:
        - If single dataset: dataset, loader
        - If ALL datasets: list of datasets, list of loaders
    """
    if flag == 'train':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
    elif flag == 'val':
        batch_size = args.batch_size
        shuffle = False
        drop_last = True
    else:  # test or pred
        batch_size = args.batch_size#1
        shuffle = False
        drop_last = False

    # 🔥 ALL datasets
    if args.data == 'ALL':
        datasets = [build_dataset(args, flag, cfg) for cfg in ALL_DATASETS]
        loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(shuffle if flag == 'train' else False),
                num_workers=0,#args.num_workers,
                drop_last=(drop_last if flag == 'train' else False)
            )
            for ds in datasets
        ]


        print(f"\n[{flag.upper()}] Dataset scaling statistics:")
        for cfg, ds in zip(ALL_DATASETS, datasets):
            name = cfg[0]
            if hasattr(ds, 'scaler') and ds.scale:
                mean = ds.scaler.mean_
                std = np.sqrt(ds.scaler.var_)
                print(f"{name:10s} | mean shape: {mean.shape}, std shape: {std.shape}")
                print(f"  mean (first 5): {mean[:5]}")
                print(f"  std  (first 5): {std[:5]}")
            else:
                print(f"{name:10s} | scaling disabled")



        return datasets, loaders

    # 🔥 Single dataset
    dataset_cfg = next(cfg for cfg in ALL_DATASETS if cfg[3] == args.data)
    dataset = build_dataset(args, flag, dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return dataset, loader


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def one_data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

