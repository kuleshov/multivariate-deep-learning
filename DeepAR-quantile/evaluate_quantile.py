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


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
      plot_batch = np.random.randint(len(test_loader)-1)
      plot_batch = 0

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      samples_list, mu_pred_list, sigma_pred_list  = [], [], [] # for gaussain
      samples_y_pred_list = [] # for quantile
      labels_list = []
      input_mu_list, input_sigma_list, input_y_pred_list  = [], [], []
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          # if i >= 1: break
          
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          #id_batch_save = id_batch
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_y_pred = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          for t in range(params.test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]

              alpha = model.get_constant_alpha([test_batch[t].shape[0], 1], 0.5)
              mu, sigma, y_pred, hidden, cell = model(test_batch[t].unsqueeze(0), alpha, id_batch, hidden, cell)
              input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              input_sigma[:,t] = v_batch[:, 0] * sigma
              input_y_pred[:,t] = v_batch[:, 0] * y_pred + v_batch[:, 1]

          input_mu_list += [input_mu]
          input_sigma_list += [input_sigma]
          input_y_pred_list += [input_y_pred]

          if sample:
              samples_gauss, sample_mu_gauss, sample_sigma_gauss = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True, sampling_method='gaussian')
              samples_quant, sample_mu_quant, sample_sigma_quant = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True, sampling_method='quantile')
              # raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics) # uncomment
              raw_metrics = utils.update_metrics(
                raw_metrics=raw_metrics,
                input_mu=input_mu, 
                input_sigma=input_sigma, 
                sample_mu=sample_mu_gauss, 
                input_y_pred=input_y_pred, 
                sample_y_pred=sample_mu_quant, 
                labels=labels, 
                predict_start=params.test_predict_start,
                samples=samples_gauss,
                samples_y_pred=samples_quant,
                relative = params.relative_metrics
              )
          else:
              sample_mu_gauss, sample_sigma_gauss, _ = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling_method='gaussian')
              _, sample_sigma_quant, sample_y_pred_quant = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling_method='quantile')
              # raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics) # uncomment
              # raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_y_pred, labels, params.test_predict_start, relative = params.relative_metrics) # uncomment
              raw_metrics = utils.update_metrics(
                raw_metrics=raw_metrics,
                input_mu=input_mu, 
                input_sigma=input_sigma, 
                sample_mu=sample_mu_gauss, 
                input_y_pred=input_y_pred, 
                sample_y_pred=sample_y_pred_quant, 
                labels=labels, 
                predict_start=params.test_predict_start,
                relative = params.relative_metrics
              )
              samples_gauss = torch.squeeze(sample_mu_gauss, 1)
              samples_quant = torch.squeeze(sample_y_pred_quant, 1)

          samples_list += [samples_gauss]
          samples_y_pred_list += [samples_quant]
          labels_list += [labels]

          if params.save_file:
              raise NotImplementedError("Didn't test this")
              if i == 0:
                  Y_hat_mean = sample_mu.data.cpu().numpy()
                  Y_hat_sig = sample_sigma.data.cpu().numpy()
                  Y = labels.data.cpu().numpy()
                  #Z = id_batch_save.numpy()
                  #V = v.numpy()
              else:
                  Y_hat_mean = np.vstack((Y_hat_mean, sample_mu.data.cpu().numpy()))
                  Y_hat_sig = np.vstack((Y_hat_sig, sample_sigma.data.cpu().numpy()))
                  Y = np.vstack((Y, labels.data.cpu().numpy()))
                  #Z = np.hstack((Z, id_batch_save.numpy()))
                  #V = np.vstack((V, v.numpy()))
          
          if i == plot_batch:
              if sample:
                  sample_metrics_gauss = utils.get_metrics(sample_mu_gauss, labels, params.test_predict_start, samples_gauss, relative = params.relative_metrics)
                  sample_metrics_quant = utils.get_metrics(sample_mu_quant, labels, params.test_predict_start, samples_quant, relative = params.relative_metrics)
              else:
                  sample_metrics_gauss = utils.get_metrics(sample_mu_gauss, labels, params.test_predict_start, relative = params.relative_metrics)                
                  sample_metrics_quant = utils.get_metrics(sample_y_pred_quant, labels, params.test_predict_start, relative = params.relative_metrics)                
              # select 10 from samples with highest error and 10 from the rest
              top_10_nd_sample = (-sample_metrics_quant['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))

              label_plot = labels[combined_sample].data.cpu().numpy()
              predict_mu = sample_mu_gauss[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma_gauss[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              # print(plot_sigma.shape)
              # print(predict_sigma.shape)
              # print(predict_sigma[0])
              # print(samples_gauss[:,combined_sample,:].data.cpu().numpy().shape)
              # print(samples_gauss[:,combined_sample,:].data.cpu().numpy()[:,0,0])
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics_gauss.items()}
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, '%d-gauss' % plot_num, plot_metrics, sample)

              label_plot = labels[combined_sample].data.cpu().numpy()
              if sample:
                predict_mu = sample_mu_quant[combined_sample].data.cpu().numpy()
              else:
                predict_mu = sample_y_pred_quant[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma_quant[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_y_pred[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              # print(plot_sigma.shape)
              # print(predict_sigma.shape)
              # print(predict_sigma[0])
              # print(samples_quant[:,combined_sample,:].data.cpu().numpy().shape)
              # print(samples_quant[:,combined_sample,:].data.cpu().numpy()[:,0,0])
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics_quant.items()}
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, '%d-quant' % plot_num, plot_metrics, sample)

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)      # uncomment
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items()) # uncomment
      if params.save_file:
        np.save(os.path.join(params.model_dir, 'Y_hat_mean.npy'), Y_hat_mean)
        np.save(os.path.join(params.model_dir, 'Y_hat_sig.npy'), Y_hat_sig)
        np.save(os.path.join(params.model_dir, 'Y.npy'), Y)
        #np.save(os.path.join(params.model_dir, 'Z.npy'), Z)
        #np.save(os.path.join(params.model_dir, 'V.npy'), V)
      logger.info('- Full test metrics: ' + metrics_string)  # uncomment

      # if sample:
      # compute CRPS
      samples_all = np.concatenate(samples_list, axis=1) # ([sample_times, num_data, pred_steps])
      samples_all = np.transpose(samples_all, [1,2,0]) # ([num_data, pred_steps, sample_times])
      labels_all = np.concatenate(labels_list, axis=0) # ([num_data, pred_steps])
      print('CRPS-gauss:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all))
      np.save(os.path.join(params.model_dir, 'labels_all.npy'), labels_all)
      np.save(os.path.join(params.model_dir, 'samples_all_gauss.npy'), samples_all)

      samples_all = np.concatenate(samples_y_pred_list, axis=1) # ([sample_times, num_data, pred_steps])
      samples_all = np.transpose(samples_all, [1,2,0]) # ([num_data, pred_steps, sample_times])
      print('CRPS-quant:', utils.crps_from_samples(labels_all[:,params.test_predict_start:], samples_all))
      np.save(os.path.join(params.model_dir, 'samples_all_quant.npy'), samples_all)

      input_mu_all = np.concatenate(input_mu_list, axis=0) # ([num_data, pred_steps])
      input_sigma_all = np.concatenate(input_sigma_list, axis=0) # ([num_data, pred_steps])
      input_y_pred_all = np.concatenate(input_y_pred_list, axis=0) # ([num_data, pred_steps])

      np.save(os.path.join(params.model_dir, 'input_mu_all.npy'), input_mu_all)
      np.save(os.path.join(params.model_dir, 'input_sigma_all.npy'), input_sigma_all)
      np.save(os.path.join(params.model_dir, 'input_y_pred_all.npy'), input_y_pred_all)


#	sampl_m = sample_mu.data.cpu().numpy()
#	sampl_s = sample_sigma.data.cpu().numpy()
#	print(sampl_m.shape)
#	print(sampl_s.shape)
#	sampl_mu = sample_mu.data.cpu().numpy()
#	sampl_sig = sample_sigma.data.cpu().numpy()
#	print(sampl_mu.shape)
#	print(sampl_sig.shape)
#	np.save('./experiments/base_model_' + str(sampl_mu.shape[1]) + '/Y_hat_mean.npy', sampl_mu.reshape(params.num_class, sampl_mu.shape[0]*sampl_mu.shape[$
#	np.save('./experiments/base_model_' + str(sampl_sig.shape[1]) + '/Y_hat_sig.npy' , sampl_sig.reshape(params.num_class, sampl_sig.shape[0]*sampl_sig.sh$
#       if params.save_file:
#           np.save(os.path.join(params.model_dir, 'Y_hat_mean.npy'), Y_hat_mean)
#           np.save(os.path.join(params.model_dir, 'Y_hat_sig.npy'), Y_hat_sig)
#           np.save(os.path.join(params.model_dir, 'Y.npy', Y)

#	sampl = samples.data.cpu().numpy()
#	label = labels.data.cpu().numpy()
#	print('sampl.shape', sampl.shape)
#	print('label.shape', label.shape)
#	print('sampl_mu.shape', sampl_mu.shape)
#	print('sampl_sig.shape', sampl_sig.shape)
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_name,
                       plot_metrics,
                       sampling=False):

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
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})


        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f'rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f} ' \
                                f'rou10: {plot_metrics["rou10"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_name) + '.png'))
    plt.close()

if __name__ == '__main__':
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

    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
