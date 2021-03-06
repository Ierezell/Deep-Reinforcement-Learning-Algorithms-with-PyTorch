import time
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from face_alignment import FaceAlignment, LandmarksType
import torchvision
import cv2
import platform
import json
import copy
from shutil import copyfile
import glob
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import nn
from tqdm import tqdm
from settings import (DEVICE, LAYERS, LOAD_PREVIOUS,
                      LOAD_PREVIOUS_RL, MODEL, PATH_WEIGHTS_DISCRIMINATOR,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR,
                      ROOT_DATASET, HALF, PARALLEL)

mplstyle.use(['dark_background', 'fast'])


def load_models(nb_pers, load_previous_state=LOAD_PREVIOUS, model=MODEL):

    if model == "small":
        from models import Discriminator, Embedder, Generator
        print("Loading Small Models (load previous)" if LOAD_PREVIOUS
              else "Loading Small Models (no load previous)")

        embedder = Embedder()
        generator = Generator()
        discriminator = Discriminator(nb_pers)

    elif model == "big":
        from bigmodels import BigDiscriminator, BigEmbedder, BigGenerator
        print("Loading Big Models (load previous)" if LOAD_PREVIOUS
              else "Loading Big Models (no load previous)")

        embedder = BigEmbedder()
        generator = BigGenerator()
        discriminator = BigDiscriminator(nb_pers)

    if HALF:
        embedder = embedder.half()
        generator = generator.half()
        discriminator = discriminator.half()

    if PARALLEL:
        embedder = nn.DataParallel(
            embedder, device_ids=range(torch.cuda.device_count()))
        generator = nn.DataParallel(
            generator, device_ids=range(torch.cuda.device_count()))
        discriminator = nn.DataParallel(
            discriminator, device_ids=range(torch.cuda.device_count()))

    if load_previous_state:
        try:
            if PARALLEL:
                embedder.module.load_state_dict(
                    torch.load(PATH_WEIGHTS_EMBEDDER))
            else:
                embedder.load_state_dict(torch.load(PATH_WEIGHTS_EMBEDDER))
        except RuntimeError:
            if PARALLEL:
                embedder.module.load_state_dict(
                    torch.load(PATH_WEIGHTS_EMBEDDER.replace(".pt", ".bk")))
            else:
                embedder.load_state_dict(
                    torch.load(PATH_WEIGHTS_EMBEDDER.replace(".pt", ".bk")))
        except FileNotFoundError:
            print("File not found, not loading weights embedder...")
        try:
            if PARALLEL:
                generator.module.load_state_dict(
                    torch.load(PATH_WEIGHTS_GENERATOR))
            else:
                generator.load_state_dict(
                    torch.load(PATH_WEIGHTS_GENERATOR))

        except RuntimeError:
            if PARALLEL:
                generator.module.load_state_dict(
                    torch.load(PATH_WEIGHTS_GENERATOR.replace(".pt", ".bk")))
            else:
                generator.load_state_dict(
                    torch.load(PATH_WEIGHTS_GENERATOR.replace(".pt", ".bk")))
        except FileNotFoundError:
            print("File not found, not loading weights generator...")

        try:
            state_dict_discriminator = torch.load(PATH_WEIGHTS_DISCRIMINATOR)
            weight_disc = True
        except RuntimeError:
            state_dict_discriminator = torch.load(
                PATH_WEIGHTS_DISCRIMINATOR.replace(".pt", ".bk"))
            weight_disc = True
        except FileNotFoundError:
            print("File not found, not loading weights discriminator...")
            weight_disc = False

        if weight_disc:
            try:
                if PARALLEL:
                    discriminator.module.load_state_dict(
                        state_dict_discriminator)
                else:
                    discriminator.load_state_dict(
                        state_dict_discriminator)
            except RuntimeError:
                print("Pas le bon dataset, different nombre de personnes")
                print("Chargement du disc sans les embeddings ")
                state_dict_discriminator.pop("embeddings.weight")
                if PARALLEL:
                    discriminator.module.load_state_dict(state_dict_discriminator,
                                                         strict=False)
                else:
                    discriminator.load_state_dict(state_dict_discriminator,
                                                  strict=False)

    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)
    return embedder, generator, discriminator


# def load_rl_model(load_previous_state=LOAD_PREVIOUS_RL):
#     from RlModel import Policy
#     policy = Policy()
#     policy = policy.to(DEVICE)
#     policy = nn.DataParallel(
#         policy, device_ids=range(torch.cuda.device_count()))

#     if load_previous_state:
#         print("Loading Rl Policy (pretrained)" if LOAD_PREVIOUS_RL
#               else "Loading Rl Policy (no pretrained)")
#         policy.module.load_state_dict(torch.load(PATH_WEIGHTS_POLICY))
#     return policy


def load_losses():
    from losses import (adverserialLoss, contentLoss,
                        discriminatorLoss, matchLoss)
    advLoss = adverserialLoss()
    mchLoss = matchLoss()
    cntLoss = contentLoss()
    dscLoss = discriminatorLoss()

    if HALF:
        advLoss = advLoss.half()
        mchLoss = mchLoss.half()
        cntLoss = cntLoss.half()
        dscLoss = dscLoss.half()

    if PARALLEL:
        advLoss = nn.DataParallel(
            advLoss, device_ids=range(torch.cuda.device_count()))
        mchLoss = nn.DataParallel(
            mchLoss, device_ids=range(torch.cuda.device_count()))
        cntLoss = nn.DataParallel(
            cntLoss, device_ids=range(torch.cuda.device_count()))
        dscLoss = nn.DataParallel(
            dscLoss, device_ids=range(torch.cuda.device_count()))

    advLoss = advLoss.to(DEVICE)
    mchLoss = mchLoss.to(DEVICE)
    cntLoss = cntLoss.to(DEVICE)
    dscLoss = dscLoss.to(DEVICE)

    return advLoss, mchLoss, cntLoss, dscLoss


def load_layers(size=LAYERS):
    if size == "small":
        from layers import (Attention, ResidualBlock,
                            ResidualBlockDown, ResidualBlockUp)
        print("Loading small layers")
        return ResidualBlock, ResidualBlockDown, ResidualBlockUp, Attention
    elif size == "big":
        from biglayers import (BigResidualBlock, BigResidualBlockDown,
                               BigResidualBlockUp, Attention)
        print("Loading big layers")
        return (BigResidualBlock, BigResidualBlockDown,
                BigResidualBlockUp, Attention)


class CheckpointsFewShots:
    def __init__(self, len_loader):
        self.best_loss_EmbGen = 1000
        self.best_loss_Disc = 1000
        self.step_disc = 0
        self.step_EmbGen = 0
        self.save_every = len_loader//4

    def save(self, model, loss, embedder, generator, discriminator):
        loss = loss.detach()
        if model == "disc":
            self.step_disc += 1
            if loss < self.best_loss_Disc or self.step_disc > self.save_every:
                self.step_disc = 0
                tqdm.write('\n' + '-'*25 + '\n' +
                           "| Poids disc sauvegardes |" + '\n' + '-'*25)
                self.best_loss_Disc = loss
                if PARALLEL:
                    torch.save(discriminator.module.state_dict(),
                               PATH_WEIGHTS_DISCRIMINATOR.replace(".pt",
                                                                  ".bk"))
                else:
                    torch.save(discriminator.state_dict(),
                               PATH_WEIGHTS_DISCRIMINATOR.replace(".pt",
                                                                  ".bk"))
                copyfile(PATH_WEIGHTS_DISCRIMINATOR.replace(".pt", ".bk"),
                         PATH_WEIGHTS_DISCRIMINATOR)
        else:
            self.step_EmbGen += 1
            if (loss < self.best_loss_EmbGen or
                    self.step_EmbGen > self.save_every):
                self.step_EmbGen = 0
                tqdm.write("\n" + "-" * 31 + '\n' +
                           "| Poids Emb & Gen sauvegardes |" + '\n' + "-"*31)
                self.best_loss_EmbGen = loss
                if PARALLEL:
                    torch.save(embedder.module.state_dict(),
                               PATH_WEIGHTS_EMBEDDER.replace(".pt", ".bk"))
                    torch.save(generator.module.state_dict(),
                               PATH_WEIGHTS_GENERATOR.replace(".pt", ".bk"))
                else:
                    torch.save(embedder.state_dict(),
                               PATH_WEIGHTS_EMBEDDER.replace(".pt", ".bk"))
                    torch.save(generator.state_dict(),
                               PATH_WEIGHTS_GENERATOR.replace(".pt", ".bk"))
                copyfile(PATH_WEIGHTS_EMBEDDER.replace(".pt", ".bk"),
                         PATH_WEIGHTS_EMBEDDER)
                copyfile(PATH_WEIGHTS_GENERATOR.replace(".pt", ".bk"),
                         PATH_WEIGHTS_GENERATOR)


class CheckpointsRl:
    def __init__(self):
        self.losses = {"Rl": []}
        self.best_loss = 1e10
        self.last_save = 0

    def addCheckpoint(self, model, loss):
        loss = loss.detach()
        self.losses[model].append(loss)
        self.last_save += 1

    def save(self, loss, policy):
        if loss < self.best_loss or self.last_save > 100:
            self.last_save = 0
            print('\n' + '-'*20)
            print("| Poids sauvegardes |")
            print('-'*20)
            self.best_loss = loss
            torch.save(policy.module.state_dict(),  PATH_WEIGHTS_POLICY)


def visualize(self, gt_landmarks, synth_im, gt_im, *models,
              save_fig=False, name='plop', show=False):
    "-----------------------"
    # TODO Faire une vraie accuracy
    accuracy = 0.5
    "------------------------"
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), num='Mon')
    im_landmarks = gt_landmarks[0].detach().cpu().permute(1, 2, 0).numpy()
    im_synth = synth_im[0].detach().cpu().permute(1, 2, 0).numpy()
    im_gt = gt_im[0].detach().cpu().permute(1, 2, 0).numpy()

    axes[0, 0].clear()
    axes[0, 0].imshow(im_landmarks/im_landmarks.max())
    axes[0, 0].axis("off")
    axes[0, 0].set_title('Landmarks')

    axes[0, 1].clear()
    axes[0, 1].imshow(im_synth/im_synth.max())
    axes[0, 1].axis("off")
    axes[0, 1].set_title('Synthesized image')

    axes[0, 2].clear()
    axes[0, 2].imshow(im_gt/im_gt.max())
    axes[0, 2].axis("off")
    axes[0, 2].set_title('Ground truth')

    axes[1, 0].clear()
    axes[1, 0].plot(self.losses["dsc"], label='Disc loss')
    axes[1, 0].set_title('Disc loss')

    axes[1, 1].clear()
    axes[1, 1].plot(self.losses["adv"], label='Adv loss')
    axes[1, 1].plot(self.losses["mch"], label='Mch loss')
    axes[1, 1].plot(self.losses["cnt"], label='Cnt loss')
    axes[1, 1].set_title('EmbGen losses')
    axes[1, 1].legend()

    axes[1, 2].clear()
    axes[1, 2].plot(accuracy)
    axes[1, 2].set_title('Accuracy')

    for i, m in enumerate(models):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in m.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append('.'.join(n.split('.')[: -1]))
                try:
                    gradient = p.grad.cpu().detach()
                    ave_grads.append(gradient.abs().mean())
                    max_grads.append(gradient.abs().max())
                except AttributeError:
                    ave_grads.append(0)
                    max_grads.append(0)
        axes[2, i].clear()
        axes[2, i].bar(np.arange(len(max_grads)), max_grads,
                       alpha=0.5, lw=1, color="c")
        axes[2, i].bar(np.arange(len(ave_grads)), ave_grads,
                       alpha=0.7, lw=1, color="r")
        axes[2, i].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        axes[2, i].set_xticks(np.arange(len(layers)))
        axes[2, i].set_xticklabels(layers, rotation="vertical",
                                   fontsize='small')
        axes[2, i].set_xlim(left=0, right=len(ave_grads))
        axes[2, i].set_ylim(bottom=0, top=max(ave_grads)+1)
        # zoom in on the lower gradient regions
        axes[2, i].set_xlabel("Layers")
        axes[2, i].set_ylabel("average gradient")
        axes[2, i].set_title(f"{m.__class__.__name__} gradient flow")
        axes[2, i].grid(True)
        axes[2, i].legend([Line2D([0], [0], color="c", lw=4),
                           Line2D([0], [0], color="r", lw=4)],
                          ['max-gradient', 'mean-gradient'])
    # if save_fig:
    #     fig.savefig(f"{ROOT_IMAGE}{name}.png", dpi=fig.dpi)
    # fig.canvas.draw_idle()
    # fig.canvas.flush_events()
    return fig


def plot_grad_flow(fig, axes, *models):
    '''
    Plots the gradients flowing through different layers in the net
    during training.
    Can be used for checking for possible gradient
    vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow'''
    for i, m in enumerate(models):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in m.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                p.detach()
                layers.append('.'.join(n.split('.')[:-1]))
                try:
                    ave_grads.append(p.grad.abs().mean())
                except AttributeError:
                    print("No gradient for layer : ", n)
                    ave_grads.append(0)
                try:
                    max_grads.append(p.grad.abs().max())
                except AttributeError:
                    max_grads.append(0)
        axes[i].bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1,
                    color="c")
        axes[i].bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1,
                    color="b")
        axes[i].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        axes[i].set_xticks(np.arange(len(layers)))
        axes[i].set_xticklabels(layers, rotation="vertical", fontsize='small')
        axes[i].set_xlim(left=0, right=len(ave_grads))
        axes[i].set_ylim(bottom=min(ave_grads), top=max(ave_grads))
        # zoom in on the lower gradient regions
        axes[i].set_xlabel("Layers")
        axes[i].set_ylabel("average gradient")
        axes[i].set_title(f"{m.__class__.__name__} gradient flow")
        axes[i].grid(True)
        axes[i].legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)],
                       ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig.cla()
    fig.clf()
    fig.canvas.draw()
    fig.canvas.flush_events()


def make_light_dataset(path_dataset, new_path):
    for folder in glob.glob(f"{new_path}/*"):
        shutil.rmtree(folder)

    for folder in glob.glob(f"{path_dataset}/*"):
        os.mkdir(f"{new_path}/{folder.split('/')[-1]}")

        for context in glob.glob(f"{folder}/*"):
            # print(context)
            nb_files = len(glob.glob(f"{context}/*"))
            if nb_files == 1:
                dest = f"{new_path}/{'/'.join(context.split('/')[-2:])}"
                shutil.copytree(context, dest)
                break


def print_parameters(model):
    trainParamModel = sum([np.prod(p.size()) if p.requires_grad else 0
                           for p in model.parameters()])
    try:
        name = model.module.__class__.__name__
    except AttributeError:
        name = model.__class__.__name__
    print(f"Nombre de parametres {name}: {trainParamModel:,}")


def print_device(model):
    try:
        try:
            name = model.module.__class__.__name__
            device_param = next(model.module.parameters()).device
        except AttributeError:
            name = model.__class__.__name__
            device_param = next(model.parameters()).device

        print(f"{name} est sur {device_param}")
    except StopIteration:
        print(f"{name} n'as pas de parametres")


def write_landmarks_on_image(image, landmarks):
    # Machoire
    cv2.polylines(image, [np.int32(landmarks[0:17])],
                  isClosed=False, color=(0, 255, 0))
    # Sourcil Gauche
    cv2.polylines(image, [np.int32(landmarks[17:22])],
                  isClosed=False, color=(255, 0, 0))
    # Sourcil droit
    cv2.polylines(image, [np.int32(landmarks[22:27])],
                  isClosed=False, color=(255, 0, 0))
    # Nez arrete
    cv2.polylines(image, [np.int32(landmarks[27:31])],
                  isClosed=False, color=(255, 0, 255))
    # Nez narine
    cv2.polylines(image, [np.int32(landmarks[31:36])],
                  isClosed=False, color=(255, 0, 255))
    # Oeil gauche
    cv2.polylines(image, [np.int32(landmarks[36:42])],
                  isClosed=True, color=(0, 0, 255))
    # oeil droit
    cv2.polylines(image, [np.int32(landmarks[42:48])],
                  isClosed=True, color=(0, 0, 255))
    # Bouche exterieur
    cv2.polylines(image, [np.int32(landmarks[48:60])],
                  isClosed=True, color=(255, 255, 0))
    # Bouche interieur
    cv2.polylines(image, [np.int32(landmarks[60:68])],
                  isClosed=True, color=(255, 255, 0))
    return image


def get_landmarks_from_webcam(self):
    cam = cv2.VideoCapture(0)
    bad_image = True
    while bad_image:
        _, image = cam.read()
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            landmark_pts = self.face_landmarks.get_landmarks_from_image(
                image)
        image = np.zeros(image.shape, np.float32)
        try:
            landmark_pts = landmark_pts[0]
            write_landmarks_on_image(image, landmark_pts)
            landmark_tensor = transforms.ToTensor()(image)
            bad_image = False
        except TypeError:
            continue
    cam.release()
    torch.cuda.empty_cache()
    return landmark_pts, landmark_tensor.unsqueeze(0).to(DEVICE)


def load_someone():
    if platform.system() == "Windows":
        slash = "\\"
    else:
        slash = "/"
    userid = np.random.choice(glob.glob(f"{ROOT_DATASET}/*"))
    filename = np.random.choice(glob.glob(f"{userid}/*.json"))[:-5]
    itemId = torch.tensor(int(filename.split(slash)[-2][2:]))

    with open(f"{filename}.json", "r") as file:
        dict_ldmk = json.load(file, object_pairs_hook=lambda x:
                              {int(k): v for k, v in x})

    frames = list(dict_ldmk.keys())

    cvVideo = cv2.VideoCapture(f"{filename}.mp4")

    cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    _, gt_im = cvVideo.read()
    gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

    gt_ldmk = dict_ldmk[frames[0]]
    gt_ldmk_im = np.zeros(gt_im.shape, np.float32)
    gt_ldmk_im = write_landmarks_on_image(gt_ldmk_im, gt_ldmk)

    context_tensors_list = []
    for frame in frames:
        cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, ctx_img = cvVideo.read()
        ctx_img = cv2.cvtColor(ctx_img, cv2.COLOR_BGR2RGB)
        ctx_ldmk = dict_ldmk[frame]
        ctx_img = write_landmarks_on_image(ctx_img, ctx_ldmk)
        ctx_img = transforms.ToTensor()(ctx_img)
        # ctx_img = transforms.Normalize([0.485, 0.456, 0.406],
        #                                [0.229, 0.224, 0.225])(ctx_img)
        context_tensors_list.append(ctx_img)

    cvVideo.release()

    gt_im_tensor = transforms.ToTensor()(gt_im)
    # gt_im_tensor = transforms.Normalize([0.485, 0.456, 0.406],
    #                                     [0.229, 0.224, 0.225])(gt_im_tensor)
    context_tensors = torch.cat(context_tensors_list).unsqueeze(0)
    gt_im_tensor = gt_im_tensor.to(DEVICE)
    context_tensors = context_tensors.to(DEVICE)
    itemId = itemId.to(DEVICE)
    return gt_im_tensor, gt_ldmk, context_tensors, itemId
