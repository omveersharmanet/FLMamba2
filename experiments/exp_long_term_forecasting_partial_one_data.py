import random

from data_provider.data_factory import ALL_DATASETS, data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Partial_one_data(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Partial_one_data, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for cfg, loader in zip(ALL_DATASETS, vali_loader):
                dataset_name = cfg[0]
                print(f"Validating on dataset: {dataset_name}")# loop over each dataset

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -cfg[6][self.args.pred_len_case]:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :cfg[7][self.args.label_len_case], :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    #print(outputs.shape, batch_y.shape,cfg[6][self.args.pred_len_case])
                    outputs = outputs[:, -cfg[6][self.args.pred_len_case]:, f_dim:]
                    batch_y = batch_y[:, -cfg[6][self.args.pred_len_case]:, f_dim:].to(self.device)
                    
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                    total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        all_batch_loaders = []
        for cfg, loader in zip(ALL_DATASETS, train_loader):
            dataset_name = cfg[0]
            for ii, batch in enumerate(loader):  # capture i here
                all_batch_loaders.append((dataset_name, cfg, ii, batch))


        for epoch in range(self.args.train_epochs):
            #vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            random.shuffle(all_batch_loaders)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (dataset_name, cfg, ii, batch) in enumerate(all_batch_loaders):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                #dataset_name = cfg[0]
                #print(f"Training on dataset: {dataset_name}")# loop over each dataset
                #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    # training code
                    #batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    #print('batch_x, batch_y, batch_x_mark, batch_y_mark',batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                partial_start = self.args.partial_start_index
                partial_end = min(self.args.enc_in + partial_start, batch_x.shape[-1])
                batch_x = batch_x[:, :, partial_start:partial_end]
                batch_y = batch_y[:, :, partial_start:partial_end]
                
                if self.args.efficient_training:
                    _, _, N = batch_x.shape
                    index = np.stack(random.sample(range(N), N))[-self.args.enc_in:]
                    batch_x = batch_x[:, :, index]
                    batch_y = batch_y[:, :, index] 
                    
                    
                    # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -cfg[6][self.args.pred_len_case]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :cfg[7][self.args.label_len_case], :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -cfg[6][self.args.pred_len_case]:, f_dim:]
                        batch_y = batch_y[:, -cfg[6][self.args.pred_len_case]:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -cfg[6][self.args.pred_len_case]:, f_dim:]
                    batch_y = batch_y[:, -cfg[6][self.args.pred_len_case]:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                        #print('batch_x, batch_y, batch_x_mark, batch_y_mark',batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                        
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                        # print(speed)
                        # allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                        # cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                        # total = allocated_memory + cached_memory
                        # print('allocated_memory:', allocated_memory)
                        # print('cached_memory:', cached_memory)
                        # print('total:', total)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_datas, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))





        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for cfg, test_data, loader in zip(ALL_DATASETS, test_datas, test_loader):
                dataset_name = cfg[0]
                print(f"Testing on dataset: {dataset_name}")

                preds_class = []
                trues_class = []

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -cfg[6][self.args.pred_len_case]:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :cfg[7][self.args.label_len_case], :], dec_inp], dim=1).float().to(self.device)

                    # forward
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -cfg[6][self.args.pred_len_case]:, f_dim:]
                    batch_y = batch_y[:, -cfg[6][self.args.pred_len_case]:, f_dim:].to(self.device)

                    outputs_np = outputs.detach().cpu().numpy()
                    batch_y_np = batch_y.detach().cpu().numpy()

                    # inverse scale
                    if test_data.scale and self.args.inverse:
                        shape = outputs_np.shape
                        outputs_np = test_data.inverse_transform(outputs_np.squeeze(0)).reshape(shape)
                        batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)
                    #print(outputs_np.shape)
                    preds_class.append(outputs_np)
                    trues_class.append(batch_y_np)

                    # Optional: visualization
                    if i % 20 == 0:
                        input_np = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input_np.shape
                            input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                        pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, f'{dataset_name}_{i}.pdf'))

                # Convert per-dataset lists to arrays
                preds_class = np.concatenate(preds_class, axis=0)
                trues_class = np.concatenate(trues_class, axis=0)
                print(np.array(preds_class).shape,np.array(trues_class).shape)
                preds_class = np.array(preds_class).reshape(-1, outputs_np.shape[-2], outputs_np.shape[-1])
                trues_class = np.array(trues_class).reshape(-1, batch_y_np.shape[-2], batch_y_np.shape[-1])
                print(cfg[6][self.args.pred_len_case])
                # Compute and print dataset-level loss (MSE)
                #dataset_loss = np.mean((preds_class - trues_class) ** 2)
                clsmae, clsmse, clsrmse, clsmape, clsmspe = metric(preds_class, trues_class)
                print('clsmse:{}, clsmae:{}'.format(clsmse, clsmae))
                print(f"{dataset_name} Loss: {clsmse:.6f}")

                # Save per-dataset predictions and ground truth
                np.save(os.path.join(folder_path, f"{dataset_name}_pred.npy"), preds_class)
                np.save(os.path.join(folder_path, f"{dataset_name}_true.npy"), trues_class)
                preds_class = np.array(preds_class).reshape(-1)
                trues_class = np.array(trues_class).reshape(-1)
                # Append to global lists
                preds.append(preds_class)
                trues.append(trues_class)

        # Optional: overall test results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Overall test shape:', preds.shape, trues.shape)




        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    def predict(self, setting, load=False):

        test_datas, test_loader = self._get_data(flag='test')
        if load:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        preds = []
        trues = []

        folder_path = './pred_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for cfg, test_data, loader in zip(ALL_DATASETS, test_datas, test_loader):
                dataset_name = cfg[0]
                print(f"Testing on dataset: {dataset_name}")

                preds_class = []
                trues_class = []

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -cfg[6][self.args.pred_len_case]:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :cfg[7][self.args.label_len_case], :], dec_inp], dim=1).float().to(self.device)

                    # forward
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -cfg[6][self.args.pred_len_case]:, f_dim:]
                    batch_y = batch_y[:, -cfg[6][self.args.pred_len_case]:, f_dim:].to(self.device)

                    outputs_np = outputs.detach().cpu().numpy()
                    batch_y_np = batch_y.detach().cpu().numpy()

                    # inverse scale
                    if test_data.scale and self.args.inverse:
                        shape = outputs_np.shape
                        outputs_np = test_data.inverse_transform(outputs_np.squeeze(0)).reshape(shape)
                        batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)
                    #print(outputs_np.shape)
                    preds_class.append(outputs_np)
                    trues_class.append(batch_y_np)

                    # Optional: visualization
                    if i % 20 == 0:
                        input_np = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input_np.shape
                            input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                        pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, f'{dataset_name}_{i}.pdf'))

                # Convert per-dataset lists to arrays
                preds_class = np.concatenate(preds_class, axis=0)
                trues_class = np.concatenate(trues_class, axis=0)
                print(np.array(preds_class).shape,np.array(trues_class).shape)
                preds_class = np.array(preds_class).reshape(-1, outputs_np.shape[-2], outputs_np.shape[-1])
                trues_class = np.array(trues_class).reshape(-1, batch_y_np.shape[-2], batch_y_np.shape[-1])
                print(cfg[6][self.args.pred_len_case])
                # Compute and print dataset-level loss (MSE)
                #dataset_loss = np.mean((preds_class - trues_class) ** 2)
                clsmae, clsmse, clsrmse, clsmape, clsmspe = metric(preds_class, trues_class)
                print('clsmse:{}, clsmae:{}'.format(clsmse, clsmae))
                print(f"{dataset_name} Loss: {clsmse:.6f}")

                # Save per-dataset predictions and ground truth
                np.save(os.path.join(folder_path, f"{dataset_name}_pred.npy"), preds_class)
                np.save(os.path.join(folder_path, f"{dataset_name}_true.npy"), trues_class)
                preds_class = np.array(preds_class).reshape(-1)
                trues_class = np.array(trues_class).reshape(-1)
                # Append to global lists
                preds.append(preds_class)
                trues.append(trues_class)

        # Optional: overall test results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Overall test shape:', preds.shape, trues.shape)




        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
