import os
import platform
import torch
import datetime
import wandb
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
PLATFORM = platform.node()[:3]

GAMMA = 0.999

LEARNING_RATE_RL = 0.01

EPS_START = 0.9

EPS_END = 0.05

EPS_DECAY = 200

MAX_DEQUE_LANDMARKS = 1000
HALF = False
MAX_ITER_PERSON = 50
PATH_WEIGHTS_DISCRIMINATOR = "../weights/blg_small_32_5e-06_5e-05_2_8_small_big_noisy_first_True_512/Discriminator.pt"
PATH_WEIGHTS_EMBEDDER = "../weights/blg_small_32_5e-06_5e-05_2_8_small_big_noisy_first_True_512/Embedder.pt"
PATH_WEIGHTS_GENERATOR = "../weights/blg_small_32_5e-06_5e-05_2_8_small_big_noisy_first_True_512/Generator.pt"

MODEL = "small"
ROOT_DATASET = "dataset/jsonDataset"


LAYERS = "big"
LOAD_PREVIOUS = False
LOAD_PREVIOUS_RL = True
LATENT_SIZE = 512
BATCH_SIZE = 1
CONCAT = ""
PARALLEL = False
