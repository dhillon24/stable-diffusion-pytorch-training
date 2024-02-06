import os
import torch
import yaml
import copy
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from utils import *
from clip import CLIPModel, clip_loss, clip_metrics
from vqgan import VQGAN
from metrics import LPIPS, FID

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
torch.set_float32_matmul_precision('medium')

class VQGANTester:

  def __init__(self, args):
    self.args = args
    self.device = args.device
    self.model = VQGAN(args)
    self.model.to(self.device)
    self.load_weights(args.vqgan_checkpoint_path)
    self.model.eval()
    self.dir_name = None
    self.test_data = None
    self.lpips = LPIPS().to(self.device)
    self.lpips.eval()
    self.fid = FID(self.device)
    self.fid.inception.eval()
    self.fid_image_activations = None
    self.fid_decoded_image_activations = None
    self.fid_dimensions = 2048
    
  def load_weights(self, path):
    self.model.load_checkpoint(path)

  def calculate_metrics(self):

    if self.dir_name is None:
      self.dir_name = setup_directories(self.args.run_name)

    test_dataloader, self.test_data = load_dataset(self.args, with_captions=False, split=False, shuffle=False)    
    pbar = tqdm(test_dataloader)

    n = len(self.test_data)
    self.fid_image_activations = np.empty((n, self.fid_dimensions))
    self.fid_decoded_image_activations = []

    start_idx = 0

    lpips = 0.0
    samples = 0

    with torch.no_grad():
      for j, (test_images, class_index) in enumerate(pbar):
        test_images = test_images.to(self.device)
        decoded_test_images, _, _ = self.model(test_images)
        image_activations = self.fid.compute_activations(test_images)
        decoded_image_activations = self.fid.compute_activations(decoded_test_images)

        if start_idx == 0:
          dim = image_activations.shape[-1]
          self.fid_image_activations = np.empty((n, dim))
          self.fid_decoded_image_activations = np.empty((n, dim))
          
        curr_batch_size = len(test_images)  
        self.fid_image_activations[start_idx: start_idx + curr_batch_size] = image_activations.detach().cpu().numpy()
        self.fid_decoded_image_activations[start_idx: start_idx + curr_batch_size] = decoded_image_activations.detach().cpu().numpy()
        start_idx += curr_batch_size
        
        batch_lpips = self.lpips(test_images, decoded_test_images)
        pbar.set_postfix({'BatchLPIPS':batch_lpips.mean().item()})
        lpips += batch_lpips.sum().item()
        samples += len(test_images)

    lpips /= samples
    logging.info("Test set LPIPS: {:.4f}".format(lpips))

    fid = self.fid.compute_distance(self.fid_image_activations, self.fid_decoded_image_activations)
    logging.info("Test set FID: {:.4f}".format(fid))


class CLIPTester:
  
  def __init__(self, args):
    self.args = args
    self.device = args.device
    self.model = CLIPModel(args)
    self.model.to(self.device)
    self.load_weights(args.clip_checkpoint_path)
    self.model.eval()
    self.dir_name = None
    self.test_data = None
    self.dataset_image_projections = None
    self.dataset_image_indices = None
    
  def load_weights(self, path):
    self.model.load_checkpoint(path)

  def calculate_embeddings(self):

    test_dataloader, self.test_data = load_dataset(self.args, with_captions=True, split=False, shuffle=False)    
    pbar = tqdm(test_dataloader)
    dataset_image_projections = []
    dataset_image_indices = []

    test_data_accuracy = 0.0

    with torch.no_grad():
      for j, (test_images, test_annotations) in enumerate(pbar):
        test_images = test_images.to(self.device)
        test_captions = test_annotations['caption']
        test_indices = test_annotations['index']

        test_image_projections, test_text_projections = self.model.query(test_images, test_captions)
        dataset_image_projections.append(test_image_projections)
        dataset_image_indices.append(test_indices)

        test_similarity = test_text_projections @ test_image_projections.T
        test_image_accuracy, test_caption_accuracy = clip_metrics(test_similarity)
        pbar.set_postfix({'ImageAccuracy':test_image_accuracy.item(),'CaptionAccuracy':test_caption_accuracy.item()})
        test_data_accuracy += test_image_accuracy.item() * len(test_images)

    test_data_accuracy /= len(self.test_data)

    logging.info("Test set accuracy: {:.4f}".format(test_data_accuracy))
    dataset_image_projections = torch.cat(dataset_image_projections, dim=0).to(self.device)
    dataset_image_indices = torch.cat(dataset_image_indices, dim=0).to(self.device)

    return dataset_image_projections, dataset_image_indices

  def query_captions(self, input):

    if self.dir_name is None:
      self.dir_name = setup_directories(self.args.run_name)

    if self.test_data is None:
      self.dataset_image_projections, self.dataset_image_indices = self.calculate_embeddings()

    with torch.no_grad():
      match_indices = self.model.find_matches(input, self.dataset_image_projections, max_matches=self.args.max_matches_test, num_captions=1)
      
      retrieval_table = []
      for k in range(len(match_indices)):
        retrieval_table.append({
          'query_filepath': self.test_data.samples[self.dataset_image_indices[k].item()][0],
          'query_caption': input[k],
          'retrieved_filepaths': [self.test_data.samples[self.dataset_image_indices[j].item()][0] for j in match_indices[k]], 
          'retrieved_captions': [self.test_data[self.dataset_image_indices[j].item()][1]['caption'] for j in match_indices[k]]
        })

      clip_plot_images(retrieval_table, save_plot=True, show_figure=False, query_image_available=False,
                        save_path=os.path.join("results", self.args.run_name, self.dir_name, f"test_results.jpg"), 
                        max_matches=self.args.max_matches_test)
  
def launch():
  config_parser = ArgumentParser()
  config_parser.add_argument("--config", help="Path to config file")
  config_args = config_parser.parse_args()
  
  with open(config_args.config, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
    for key,value in config.items():
      config_parser.add_argument(f"--{key}", type=type(value), default=value)
      
  args = config_parser.parse_args()
  
  if args.run_name == "CLIP":
    tester = CLIPTester(args)
    tester.query_captions(['a cat playing in the snow', 'a fighter jet in the sky', 'a black horse standing'])
  elif args.run_name == "VQGAN":
    tester = VQGANTester(args)
    tester.calculate_metrics()

if __name__ == '__main__':
  launch()