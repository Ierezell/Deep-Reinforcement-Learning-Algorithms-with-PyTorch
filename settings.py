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

MAX_DEQUE_LANDMARKS = 10000
HALF = False
MAX_ITER_PERSON = 5000
PATH_WEIGHTS_DISCRIMINATOR = "./weights/top_big/Discriminator.pt"
PATH_WEIGHTS_EMBEDDER = "./weights/top_big/Embedder.pt"
PATH_WEIGHTS_GENERATOR = "./weights/top_big/Generator.pt"

MODEL = "big"
ROOT_DATASET = "dataset/jsonDataset"


LAYERS = "big"
LOAD_PREVIOUS = True
LOAD_PREVIOUS_RL = True
LATENT_SIZE = 512
BATCH_SIZE = 1
CONCAT = ""
PARALLEL = False
