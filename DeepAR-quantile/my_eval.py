import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-file', action='store_true', help='Whether to save values during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'



# Load the parameters
args = parser.parse_args()
model_dir = os.path.join('experiments', args.model_name) 
json_path = os.path.join(model_dir, 'params.json')
data_dir = model_dir #os.path.join(args.data_folder, args.dataset) changed !!!!!!!!!!!!!!!!!!
assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
params = utils.Params(json_path)

utils.set_logger(os.path.join(model_dir, 'eval.log'))

params.relative_metrics = args.relative_metrics
params.sampling = args.sampling
params.model_dir = model_dir
params.save_file = args.save_file
params.plot_dir = os.path.join(model_dir, 'figures')

LOOKAHEAD = 24
params.test_window = min(params.test_window, params.test_predict_start+LOOKAHEAD)

cuda_exist = torch.cuda.is_available()  # use GPU is available

# Set random seeds for reproducible experiments if necessary
if cuda_exist:
    params.device = torch.device('cuda')
    # torch.cuda.manual_seed(240)
    logger.info('Using Cuda...')
    model = net.Net(params).cuda()
else:
    params.device = torch.device('cpu')
    # torch.manual_seed(230)
    logger.info('Not using cuda...')
    model = net.Net(params)

# Create the input data pipeline
logger.info('Loading the datasets...')

test_set = TestDataset(data_dir, args.dataset, params.num_class)
test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=SequentialSampler(test_set), num_workers=1)
##RandomSampler
logger.info('Sampler iteration')
# print([i for i in SequentialSampler(test_set).__iter__()])
logger.info('- done.')

print('model: ', model)
loss_fn = net.loss_fn

logger.info('Starting evaluation')

# Reload weights from the saved file
utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

# Load saved data
input_mu_all = np.load(os.path.join(params.model_dir, 'input_mu_all.npy'))
input_sigma_all = np.load(os.path.join(params.model_dir, 'input_sigma_all.npy'))
input_y_pred_all = np.load(os.path.join(params.model_dir, 'input_y_pred_all.npy'))

# load saved samples # ([num_data, pred_steps, sample_times])
labels_all = np.load(os.path.join(params.model_dir, 'labels_all.npy'))
samples_all_gauss = np.load(os.path.join(params.model_dir, 'samples_all_gauss.npy'))
samples_all_quant = np.load(os.path.join(params.model_dir, 'samples_all_quant.npy'))

# truncate if necessary
labels_all = labels_all[:,:params.test_predict_start+LOOKAHEAD]
samples_all_gauss = samples_all_gauss[:,:LOOKAHEAD,:]
samples_all_quant = samples_all_quant[:,:LOOKAHEAD,:]

# compute CRPS
print('CRPS-gauss:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_gauss))
print('CRPS-quant:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_quant))

# compute effective predicted quantiles
quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# quantiles_gauss has shape ([num_quantiles, num_data, pred_steps])
quantiles_gauss = np.quantile(samples_all_gauss, quantiles, axis=2)
quantiles_quant = np.quantile(samples_all_quant, quantiles, axis=2)

# compute calibration
pred_steps = samples_all_gauss.shape[1]
labels_all_expanded = labels_all[np.newaxis, :, params.test_predict_start:] # (1, num_data, pred_steps)
quantile_indicators_gauss = labels_all_expanded <= quantiles_gauss
quantile_indicators_quant = labels_all_expanded <= quantiles_quant
calibration_gauss = quantile_indicators_gauss.mean(axis=2).mean(axis=1)
calibration_quant = quantile_indicators_quant.mean(axis=2).mean(axis=1)
# calibration_counts_gauss = quantile_indicators_gauss.sum(axis=2).sum(axis=1)
# calibration_counts_quant = quantile_indicators_quant.sum(axis=2).sum(axis=1)
print(quantiles)
print(calibration_gauss, np.abs(calibration_gauss-quantiles).mean())
print(calibration_quant, np.abs(calibration_quant-quantiles).mean())
print()

print(quantiles)
for i in range(3):
  print(quantile_indicators_gauss.mean(axis=2)[:,i])
  print(quantile_indicators_quant.mean(axis=2)[:,i])

# visualize some time series
def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_top,
                       predict_bottom,
                       labels,
                       samples,
                       window_size,
                       predict_start,
                       plot_name):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_bottom[m, :],
                         predict_top[m, :], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')
        # create scatter plot
        # points = np.concatenate([
        #   x[predict_start:, np.newaxis, np.newaxis], 
        #   samples[:, :, np.newaxis]
        # ], axis=2)

    f.savefig(os.path.join(plot_dir, str(plot_name) + '.png'))
    plt.close()

# batch_size = params.predict_batch
# sample_mu_gauss = np.median(samples_all_gauss[:batch_size], axis=2)
# sample_metrics_gauss = utils.get_metrics(
#   torch.from_numpy(np.median(samples_all_gauss[:batch_size], axis=2)), 
#   torch.from_numpy(labels_all[:batch_size]), params.test_predict_start, 
#   torch.from_numpy(samples_all_gauss[:batch_size].transpose([2,0,1])))
# sample_metrics_quant = utils.get_metrics(
#   torch.from_numpy(np.median(samples_all_quant[:batch_size], axis=2)), 
#   torch.from_numpy(labels_all[:batch_size]), params.test_predict_start, 
#   torch.from_numpy(samples_all_quant[:batch_size].transpose([2,0,1])))

# # select 10 from samples with highest error and 10 from the rest
# top_10_nd_sample = (-sample_metrics_quant['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
# chosen = set(top_10_nd_sample.tolist())
# all_samples = set(range(batch_size))
# not_chosen = np.asarray(list(all_samples - chosen))
# random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
# random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
# combined_sample = np.concatenate((random_sample_10, random_sample_90))    

# plot_mean = np.concatenate([input_mu_all[combined_sample], quantiles_gauss[4,combined_sample,:]], axis=1)
# plot_eight_windows(
#   '.', plot_mean, quantiles_gauss[-1,combined_sample,:], quantiles_gauss[0,combined_sample,:], labels_all[combined_sample], 
#   params.test_window, params.test_predict_start, 'plots-gauss'
# )

# plot_mean = np.concatenate([input_mu_all[combined_sample], quantiles_quant[4,combined_sample,:]], axis=1)
# plot_eight_windows(
#   '.', plot_mean, quantiles_quant[-1,combined_sample,:], quantiles_quant[0,combined_sample,:], labels_all[combined_sample], 
#   params.test_window, params.test_predict_start, 'plots-quant'
# )

np.random.seed(0)
# random_sample = np.concatenate([
#   np.random.choice(range(1000), size=10, replace=False),
#   np.random.choice(range(samples_all_gauss.shape[0]), size=10, replace=False)
# ])
random_sample = np.random.choice(range(samples_all_gauss.shape[0]), size=20, replace=False)

plot_mean = np.concatenate([input_mu_all, quantiles_gauss[4,:,:]], axis=1)
plot_eight_windows(
  '.', plot_mean[random_sample], quantiles_gauss[-1,random_sample,:], 
  quantiles_gauss[0,random_sample,:], labels_all[random_sample], samples_all_gauss[random_sample],
  params.test_window, params.test_predict_start, 'plots-gauss'
)

plot_mean = np.concatenate([input_mu_all, quantiles_quant[4,:,:]], axis=1)
plot_eight_windows(
  '.', plot_mean[random_sample], quantiles_quant[-1,random_sample,:], 
  quantiles_quant[0,random_sample,:], labels_all[random_sample], samples_all_gauss[random_sample],
  params.test_window, params.test_predict_start, 'plots-quant'
)


# compute CRPS
samples_all_gauss_bak = samples_all_gauss.copy()
samples_all_quant_bak = samples_all_quant.copy()
labels_all_bak = labels_all.copy()
crps_gauss, crps_quant, cal_err_gauss, cal_err_quant = [], [], [], []
random_sample = np.random.choice(range(samples_all_gauss.shape[0]), size=10000, replace=False)
# for i in range(samples_all_gauss_bak.shape[0]):
for i in random_sample:
  samples_all_gauss = samples_all_gauss_bak[[[i]]]
  samples_all_quant = samples_all_quant_bak[[i]]
  labels_all = labels_all_bak[[i]]
  crps_gauss += [utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_gauss)]
  crps_quant += [utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_quant)]
  # print(i, 'CRPS-gauss:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_gauss))
  # print(i, 'CRPS-quant:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all_quant))

  # compute effective predicted quantiles
  quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
  # quantiles_gauss has shape ([num_quantiles, num_data, pred_steps])
  quantiles_gauss = np.quantile(samples_all_gauss, quantiles, axis=2)
  quantiles_quant = np.quantile(samples_all_quant, quantiles, axis=2)

  # compute calibration
  pred_steps = samples_all_gauss.shape[1]
  labels_all_expanded = labels_all[np.newaxis, :, params.test_predict_start:] # (1, num_data, pred_steps)
  quantile_indicators_gauss = labels_all_expanded <= quantiles_gauss
  quantile_indicators_quant = labels_all_expanded <= quantiles_quant
  calibration_gauss = quantile_indicators_gauss.mean(axis=2).mean(axis=1)
  calibration_quant = quantile_indicators_quant.mean(axis=2).mean(axis=1)
  cal_err_gauss += [np.abs(calibration_gauss-quantiles).mean()]
  cal_err_quant += [np.abs(calibration_quant-quantiles).mean()]
  # print(quantiles)
  # print(calibration_gauss)
  # print(calibration_quant)
  # print()

  # # print(quantiles)
  # # for i in range(3):
  # #   print(quantile_indicators_gauss.mean(axis=2)[:,i])
  # #   print(quantile_indicators_quant.mean(axis=2)[:,i])

crps_quant = np.array(crps_quant)
crps_gauss = np.array(crps_gauss)
crps_avrge = (crps_gauss + crps_quant)/2
crps_diffr = crps_gauss - crps_quant
print('Same:', np.sum((np.abs(crps_diffr) / crps_avrge) <= 0.05))
print('Gauss better:', np.sum((crps_diffr / crps_avrge) < -0.05))
print('Quant better:', np.sum((crps_diffr / crps_avrge) > 0.05))

cal_err_quant = np.array(cal_err_quant)
cal_err_gauss = np.array(cal_err_gauss)
cal_err_avrge = (cal_err_gauss + cal_err_quant)/2
cal_err_diffr = cal_err_gauss - cal_err_quant
print('Same:', np.sum((np.abs(cal_err_diffr) / cal_err_avrge) <= 0.10))
print('Gauss better:', np.sum((cal_err_diffr / cal_err_avrge) < -0.10))
print('Quant better:', np.sum((cal_err_diffr / cal_err_avrge) > 0.10))