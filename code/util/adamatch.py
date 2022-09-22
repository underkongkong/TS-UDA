import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    correct_k = correct.reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)

def train_ada(model, source_weak_dataloader, source_strong_dataloader, target_weak_dataloader, target_strong_dataloader,
          source_criterion, target_criterion, optimizer, scheduler, device, epoch):
        dataset_loader = zip(source_weak_dataloader, source_strong_dataloader, target_weak_dataloader, target_strong_dataloader)
        epoch_start = time.time()
        trn_losses = AverageMeter()
        adjust_losses = AverageMeter()
        src_trn_acc = AverageMeter()
        tgt_trn_acc = AverageMeter()
        steps_per_epoch = min(len(source_weak_dataloader), len(target_weak_dataloader))
        
        URATIO = 3
        BATCH_SIZE = 16
        NCLASSES = 5
        TAU = 0.9
        EPOCHS = 20
        total_steps = EPOCHS * steps_per_epoch
        model.train()
        tq = tqdm(enumerate(dataset_loader), total=steps_per_epoch)
        for n_step, ((data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, labels_target), (data_target_strong, _)) in tq:
            data_source_weak, labels_source, data_source_strong = data_source_weak.to(device), labels_source.to(device), data_source_strong.to(device)
            data_target_weak, labels_target, data_target_strong = data_target_weak.to(device), labels_target.to(device), data_target_strong.to(device)

            assert data_source_weak.size(0) * URATIO == data_target_weak.size(0)
            
            source_batch = torch.cat([data_source_weak, data_source_strong])
            all_batch = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong])

            optimizer.zero_grad()
            all_logits = model(all_batch)
            all_logits_source = all_logits[:BATCH_SIZE*2]

            model.eval()
            source_logits = model(source_batch)
            model.train()

            # Random logit interpolation
            lam = torch.rand(BATCH_SIZE*2, NCLASSES).to(device)
            logits_source = lam * all_logits_source + (1-lam) * source_logits

            # Distribution alignment
            logits_source_weak = logits_source[:BATCH_SIZE]
            pseudo_source = F.softmax(logits_source_weak, 0)

            logits_target = all_logits[BATCH_SIZE*2:]
            logits_target_weak = logits_target[:BATCH_SIZE*URATIO]
            pseudo_target = F.softmax(logits_target_weak, 0)

            assert pseudo_source.shape == (BATCH_SIZE, NCLASSES) and pseudo_target.shape == (URATIO*BATCH_SIZE, NCLASSES)

            expect_ratio = torch.mean(pseudo_source) / torch.mean(pseudo_target)
            final_pseudolabels = F.normalize(pseudo_target*expect_ratio)

            # Relative confidence threshold
            pseudo_source_max = torch.max(pseudo_source, dim=1)[0]
            c_tau = TAU * torch.mean(pseudo_source_max, 0)

            final_pseudolabels_max, final_pseudolabels_cls = torch.max(final_pseudolabels, dim=1)
            mask = final_pseudolabels_max >= c_tau

            source_loss = source_criterion(logits_source_weak ,labels_source) + source_criterion(logits_source[BATCH_SIZE:] ,labels_source)
            pseudolabels = final_pseudolabels_cls.detach() #stop_gradient()
            target_loss = torch.mean(target_criterion(logits_target[URATIO*BATCH_SIZE:], pseudolabels) * mask, 0)
            
            PI = torch.tensor(np.pi).to(device)
            mu = 0.5 - torch.cos(torch.minimum(PI, (2*PI*(n_step+steps_per_epoch*(epoch))) / total_steps)) / 2
            total_loss = source_loss + mu * target_loss

            acc_s = accuracy(logits_source_weak ,labels_source) 
            acc_t = accuracy(final_pseudolabels ,labels_target) 
            trn_losses.update(total_loss.item(), data_source_weak.size(0))
            adjust_losses.update(total_loss.item() / (1+mu), data_source_weak.size(0))
            src_trn_acc.update(acc_s.item(), data_source_weak.size(0))
            tgt_trn_acc.update(acc_t.item(), data_target_weak.size(0))

            total_loss.backward()
            optimizer.step()

            if (n_step+1) % 100 == 0:
                tq.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Adj Loss: {:.4f}'.format(
                epoch+1, EPOCHS, n_step+1, steps_per_epoch, trn_losses.avg, adjust_losses.avg))

        print(f'----------------Epoch {epoch+1} train finished------------------')
        print('Epoch [{}/{}], Time elapsed: {:.4f}s, Loss: {:.4f}, Adj Loss: {:.4f}, Source acc: {:.4f}%, Target acc: {:.4f}%, learning rate: {:.6f}, mu: {:.6f}'.format(
                epoch+1, EPOCHS, time.time()-epoch_start, trn_losses.avg, adjust_losses.avg, src_trn_acc.avg, tgt_trn_acc.avg, optimizer.param_groups[0]["lr"], mu))
        scheduler.step()
        return trn_losses.avg, adjust_losses.avg, src_trn_acc.avg, tgt_trn_acc.avg, mu

def evaluate_ada(model, data_loader, device):
    eval_start = time.time()
    # test_acc = AverageMeter()
    labels_list = list()
    outputs_list = list()
    preds_list = list()
    model.eval()
    tq = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
      for _, (data, labels) in tq:
        data, labels = data.to(device) ,labels.to(device)

        outputs = F.softmax(model(data), dim=1)

        labels_numpy = labels.detach().cpu().numpy()
        outputs_numpy = outputs.detach().cpu().numpy() 
        preds = np.argmax(outputs_numpy, axis=1) # accuracy

        labels_list.append(labels_numpy)
        outputs_list.append(outputs_numpy)
        preds_list.append(preds)

    labels_list = np.concatenate(labels_list)
    outputs_list = np.concatenate(outputs_list)
    preds_list = np.concatenate(preds_list)
    accuracy = accuracy_score(labels_list, preds_list)
    tq.write(f'Evaluation finished with time {time.time()-eval_start}s, accuracy: {accuracy*100}%')
    return accuracy, labels_list, outputs_list, preds_list