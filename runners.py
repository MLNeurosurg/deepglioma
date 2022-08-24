import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import custom_replace, custom_mask_missing
import matplotlib.pyplot as plt


def _easy_image_viewer(images):
    img = images[0, :, :, :]
    img = img.swapaxes(0, -1)
    plt.imshow(img)
    plt.show()


def _batch_label_weights(labels, weights):
    """Get label weight tensor for each batch."""
    batch_weights = torch.zeros(labels.shape).cuda()
    for i in range(weights.shape[1]):
        batch_weights[:, i] = torch.where(labels[:, i] == 1, weights[1, i],
                                          weights[0, i])
    return batch_weights


# runner for supervised training
def run_epoch(model,
              data,
              optimizer=None,
              scheduler=None,
              n_labels=3,
              label_weights=None,
              iterations=1000,
              train=True,
              model_type='linear'):

    if train:
        model.train()
    else:
        model.eval()

    # initialize vectors to store results
    if train:
        n_preds = int(iterations * data.batch_size)
        all_predictions = torch.zeros(n_preds, n_labels).cpu()
        all_targets = torch.zeros(n_preds, n_labels).cpu()
        all_masks = torch.zeros(n_preds, n_labels).cpu()
        all_missing_masks = torch.zeros(n_preds, n_labels).cpu()
        all_image_ids = []
    else:
        all_predictions = torch.zeros(len(data.dataset), n_labels).cpu()
        all_targets = torch.zeros(len(data.dataset), n_labels).cpu()
        all_masks = torch.zeros(len(data.dataset), n_labels).cpu()
        all_missing_masks = torch.zeros(len(data.dataset), n_labels).cpu()
        all_image_ids = []

    # training loop
    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0
    correct = 0
    total = 0
    for batch in data:

        labels = batch['labels'].float()
        images = batch['image'].float()
        all_image_ids += batch['imageIDs']
        # _easy_image_viewer(images)

        # masks for label mask training
        mask = batch['mask'].float()
        unk_mask = custom_replace(mask, 1, 0, 0)
        mask_in = mask.clone()

        # mask for missing labels, PRESENT VALUES ARE 1, MISSING VALUES 0
        missing_mask = custom_mask_missing(labels)

        # train or test
        if train:
            preds, _ = model(images.cuda(), mask_in.cuda())

        else:
            with torch.no_grad():
                preds, _ = model(images.cuda(), mask_in.cuda())

        # compute loss
        loss = F.binary_cross_entropy_with_logits(preds.view(
            labels.size(0), -1),
            labels.cuda(),
            reduction='none')

        # apply label weights if needed
        if label_weights is not None:
            label_weights = torch.tensor(label_weights).cuda()
            loss *= _batch_label_weights(labels.cuda(), label_weights)

        if model_type is 'tran':
            loss_out = (unk_mask.cuda() * missing_mask.cuda() * loss).sum()
        else:
            loss_out = (missing_mask.cuda() * loss).sum()

        if train:
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()

        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx, end_idx = (batch_idx * data.batch_size), ((batch_idx + 1) *
                                                             data.batch_size)

        if preds.size(0) != all_predictions[start_idx:end_idx].size(0):
            preds = preds.view(labels.size(0), -1)

        # append values
        # pred = pred.squeeze()
        all_predictions[start_idx:end_idx] = preds.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        all_missing_masks[start_idx:end_idx] = missing_mask.data.cpu()

        # online batch statistics
        batch_idx += 1
        preds = torch.ge(torch.sigmoid(preds), 0.5).float()
        preds = preds.cpu().numpy()
        missing_mask = missing_mask.cpu().numpy()
        labels = labels.cpu().numpy()
        # mask the missing labels
        labels = np.ma.masked_array(labels, mask=~missing_mask.astype(bool))
        preds = np.ma.masked_array(preds, mask=~missing_mask.astype(bool))
        # compute batch accuracy
        correct += (preds == labels).sum()
        total += (labels.shape[0] * labels.shape[1])
        print("Iteration: " + str(batch_idx) + " >>>> epoch accuracy: " +
              str(correct / total),
              end='\r',
              flush=True)

        # break at final iteration
        if train and batch_idx >= iterations:
            break

    if train:
        scheduler.step()
    loss_total = loss_total / float(all_predictions.size(0))
    unk_loss_total = unk_loss_total / float(all_predictions.size(0))

    return all_predictions, all_targets, all_masks, all_missing_masks, all_image_ids, loss_total, unk_loss_total


# runner for patch contrastive learning
def run_epoch_patchcon(model,
                     data,
                     supcon_loss,
                     optimizer,
                     scheduler=None,
                     n_labels=3,
                     label_weights=None,
                     iterations=1000,
                     train=True):

    if train:
        model.train()
    else:
        model.eval()

    # training loop
    iter_idx = 0
    all_image_ids = []
    epoch_losses = []
    loss_total = 0
    for batch in data:

        labels = batch['labels'].float()
        images = batch['image'].float()
        all_image_ids += batch['imageIDs']
        # _easy_image_viewer(images)

        # train or test
        if train:
            optimizer.zero_grad()
            preds, _ = model(images.cuda())
        else:
            with torch.no_grad():
                preds, _ = model(images.cuda())

        # compute loss
        losses = torch.zeros(n_labels).cuda()
        for i in range(n_labels):
            losses[i] = supcon_loss(preds[i], labels[:, i].cuda())

        if label_weights is not None:
            losses *= label_weights.cuda().view(-1)

        # optimizer update
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Updates ##
        loss_total += loss.item()
        epoch_losses.append(loss.detach().cpu().numpy())

        for i in range(n_labels):
            print(f"Iteration {iter_idx}: epoch loss >>> {loss}",
                  end='\r',
                  flush=True)

        iter_idx += 1
        # break at final iteration
        if train:
            if iter_idx >= iterations:
                break
    if train:
        scheduler.step()

    print(loss_total / iterations)
    return loss_total / iterations, epoch_losses, all_image_ids


# runner to get features
def run_epoch_features(model, data, n_labels=3, feature_dim=2048, device='cpu'):

    model.eval()

    # training loop
    iter_idx = 0
    all_predictions = torch.zeros(len(data.dataset), n_labels).cpu()
    all_features = torch.zeros(len(data.dataset), feature_dim).cpu()
    all_targets = torch.zeros(len(data.dataset), n_labels).cpu()
    all_image_ids = []
    correct = 0
    total = 0
    for batch in data:

        labels = batch['labels'].float()
        images = batch['image'].float()
        all_image_ids += batch['imageIDs']

        with torch.no_grad():
            preds, features = model(images.to(device))

        preds = torch.ge(torch.sigmoid(preds), 0.5).float()
        preds = preds.cpu()
        labels = labels.cpu()

        # compute batch accuracy
        correct += (preds == labels).sum()
        total += (labels.shape[0] * labels.shape[1])
        print("Iteration: " + str(iter_idx) + " >>>> epoch accuracy: " +
              str(correct / total),
              end='\r',
              flush=True)

        start_idx, end_idx = (iter_idx * data.batch_size), ((iter_idx + 1) *
                                                            data.batch_size)

        if preds.size(0) != all_predictions[start_idx:end_idx].size(0):
            preds = preds.view(labels.size(0), -1)
        if features.size(0) != all_predictions[start_idx:end_idx].size(0):
            features = features.view(labels.size(0), -1)

        # pred = pred.squeeze()
        all_predictions[start_idx:end_idx] = preds.data.cpu()
        all_features[start_idx:end_idx] = features.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()

        # increment index
        iter_idx += 1

    return all_image_ids, all_predictions, all_targets, all_features
