from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

viz = Visualizer(env='rtfm', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ckpt_path = os.path.join(args.output_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    
    gt = np.load(args.gt)

    # auc = test(test_loader, model, args, viz, device, gt)

    print("Using CUDA:", next(model.parameters()).is_cuda)

    for epoch in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if epoch > 1 and config.lr[epoch - 1] != config.lr[epoch - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[epoch - 1]

        if (epoch - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (epoch - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if epoch % 5 == 0 and epoch > 200:

            auc = test(test_loader, model, args, viz, device, gt)
            test_info["epoch"].append(epoch)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), os.path.join(ckpt_path, args.model_name + '{}-i3d.pkl'.format(epoch)))
                save_best_record(test_info, os.path.join(args.output_path, '{}-epoch-AUC.txt'.format(epoch)))
    torch.save(model.state_dict(), os.path.join(ckpt_path, args.model_name + 'final.pkl'))

