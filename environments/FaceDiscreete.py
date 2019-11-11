
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from torchvision import transforms

from settings import (DEVICE, MAX_DEQUE_LANDMARKS, MAX_ITER_PERSON, MODEL,
                      ROOT_DATASET)
from utils import load_models, write_landmarks_on_image, load_someone
import glob
import gym
from gym import spaces
from gym.utils import seeding

plt.ion()


class FaceEnvironementDiscreete(gym.Env):
    environment_name = "FaceEnv"

    def __init__(self, path_weights):
        super(FaceEnvironementDiscreete, self).__init__()
        # 0 Nothing / 1 Left / 2 Right / 3 Up / 4 Down
        self.id = "FaceEnv"
        self.action_space = spaces.Discrete(68 * 4)
        self.observation_space = spaces.Discrete(68 * 2)
        self._max_episode_steps = MAX_ITER_PERSON
        self.seed()

        (self.embedder,
         self.generator,
         self.discriminator) = load_models(3000)
        self.embedder = self.embedder.eval()
        self.generator = self.generator.eval()
        self.discriminator = self.discriminator.eval()

        self.landmarks = None
        self.landmarks_done = deque(maxlen=MAX_DEQUE_LANDMARKS)

        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.layersUp = None

        self.iterations = 0
        self.episodes = -1
        self.state = None

        self.fig, self.axes = plt.subplots(2, 2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def new_person(self):
        torch.cuda.empty_cache()
        self.landmarks_done = deque(maxlen=MAX_DEQUE_LANDMARKS)

        (_,
         self.landmarks,
         self.contexts,
         self.user_ids) = load_someone()
        self.landmarks = np.array(self.landmarks).flatten()
        with torch.no_grad():
            (self.embeddings,
             self.paramWeights,
             self.paramBias,
             self.layersUp) = self.embedder(self.contexts)

        self.iterations = 0
        self.episodes += 1
        self.synth_im = self.contexts.narrow(1, 0, 3)
        synth_im = self.synth_im.squeeze().cpu().permute(1, 2, 0).numpy()
        synth_im -= synth_im.min()

        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(synth_im/synth_im.max())
        self.axes[0, 0].axis("off")
        self.axes[0, 0].set_title('State')

        self.axes[1, 0].clear()
        self.axes[1, 0].imshow(synth_im/synth_im.max())
        self.axes[1, 0].axis("off")
        self.axes[1, 0].set_title('Ref')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # torch.cuda.empty_cache()

    def step(self, action):
        self.iterations += 1
        done = False
        if self.iterations > MAX_ITER_PERSON:
            done = True

        ldmk_no = action // 68
        action_type = action % 68
        if action_type == 0:
            self.landmarks.reshape(-1, 2)[ldmk_no][0] += 5
        elif action_type == 1:
            self.landmarks.reshape(-1, 2)[ldmk_no][1] += 5
        elif action_type == 2:
            self.landmarks.reshape(-1, 2)[ldmk_no][0] -= 5
        elif action_type == 3:
            self.landmarks.reshape(-1, 2)[ldmk_no][1] -= 5

        reward = self.get_reward()
        return self.landmarks, reward, done, None

    def get_reward(self):
        landmarks_img = write_landmarks_on_image(np.zeros((224, 224, 3),
                                                          dtype=np.float32),
                                                 self.landmarks.reshape(-1, 2))

        landmarks_img = transforms.ToTensor()(landmarks_img)
        landmarks_img = landmarks_img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            self.synth_im = self.generator(landmarks_img,
                                           self.paramWeights,
                                           self.paramBias, self.layersUp)

        ldmk_im = landmarks_img.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(ldmk_im/ldmk_im.max())
        self.axes[0, 1].axis("off")
        self.axes[0, 1].set_title('Landmarks (latent space)')

        synth_im = self.synth_im.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(synth_im / synth_im.max())
        self.axes[0, 0].axis("off")
        self.axes[0, 0].set_title('State')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # print("self.synth_im", self.synth_im.size())
        # print("self.landmarks_img", self.landmarks_img.size())
        # print(" self.user_ids", self.user_ids.size())
        with torch.no_grad():
            score_disc, _ = self.discriminator(torch.cat((self.synth_im,
                                                          landmarks_img),
                                                         dim=1),
                                               self.user_ids)
        score_disc = float(score_disc.data.cpu().numpy())

        if list(self.landmarks) in [list(ld) for ld in self.landmarks_done]:
            score_redoing = -100
        else:
            score_redoing = 0
            self.landmarks_done.append(self.landmarks)

        # print("score_disc : ", score_disc)
        # print("score_redoing : ", score_redoing)
        # print("score_outside : ", score_outside)
        # print("Score Tot : ", score_disc/10 + score_redoing + score_outside)
        # print("\n")
        reward = score_disc / 10 + score_redoing
        return reward

    def reset(self):
        self.landmarks_done = deque(maxlen=1000)
        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.layersUp = None
        self.iterations = 0
        self.episodes = 0
        self.new_person()

        return self.landmarks
        # torch.cuda.empty_cache()

    def reset_environment(self):
        return self.reset()

    def finish(self):
        # self.writer.close()
        # torch.cuda.empty_cache()
        pass
