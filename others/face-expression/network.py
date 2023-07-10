import os
import numpy as np
import torch
import torch.nn as nn
from model.resnet import resnet18
from model.alexnet import alexnet
from torch.nn import init
import pdb

def create_model(opt):
  model = resnet18(pretrained=False)
  #model.conv1.eval()
  #model.layer1.eval()
  #model.layer2.eval()
  #model.layer2_expression.eval()
  #model.layer3_expression.eval()
  #model.layer4_expression.eval()
  #model.layer2_age.eval()
  #model.layer3_age.eval()
  #model.layer4_age.eval()
  #model.layer2_gender.eval()
  #model.layer3_gender.eval()
  #model.layer4_gender.eval()
  #model.fc_expression.eval()
  #model.fc_age.eval()
  #model.fc_gender.eval()
  return model

class Face_Expression(nn.Module):
  def __init__(self, opt):
    super(Face_Expression, self).__init__()
    self.model = create_model(opt)
    #self.age_regress = nn.Linear(2048, 1)

  def forward(self, x):
    #feat, cls_expression_prob, cls_gender_prob, cls_age_prob = self.model(x)
    #feat = self.model(x)
    expression,age,gender,feat_expression,feat_gender,feat_age=self.model(x)
    return expression,age,gender,feat_expression,feat_gender,feat_age


class Triplet_Semi_Hard_Loss(nn.Module):
  def __init__(self):
    super(Triplet_Semi_Hard_Loss, self).__init__()
    self.margin = 0.3

  def pairwise_distance_torch(self, embeddings):

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings.mm(precise_embeddings.transpose(0, 1))

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).cuda())
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.cuda(), mask_offdiagonals.cuda())
    return pairwise_distances

  def TripletSemiHardLoss(self, y_true, y_pred, margin=1.0):
    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = self.pairwise_distance_torch(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).cuda()
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).cuda())).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss

  def forward(self, input, target):
    n = input.size(0)
    input = input.view(n,-1)
    target = torch.reshape(target, [n,1])
    input = 1.0 * input / (torch.norm(input, 2, 1, keepdim=True) + 1e-12)
    return self.TripletSemiHardLoss(target, input, margin = self.margin)
