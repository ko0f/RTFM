import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import os

'''
Feeds all test features of all test files into our model, then compares predicted Y with our ground truth Y array.

Creates a ROC curve based on comparison between those two above.
'''

def test(dataloader, model, args, viz, device, gt):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save(os.path.join(args.output_path, 'fpr.npy'), fpr)
        np.save(os.path.join(args.output_path, 'tpr.npy'), tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save(os.path.join(args.output_path, 'precision.npy'), precision)
        np.save(os.path.join(args.output_path, 'recall.npy'), recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc

