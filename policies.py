from torch import nn
from collections import deque
import torch
from settings import MAX_DEQUE_LANDMARKS


class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden = 512
        self.fc1 = nn.Linear(self.state_size, self.hidden, bias=True)
        self.rnn = nn.LSTM(input_size=self.hidden, hidden_size=self.hidden,
                           num_layers=2, bias=True, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.fc2 = nn.Linear(self.hidden*2, self.action_size, bias=True)
        self.prev_repr_ldmks = deque(maxlen=MAX_DEQUE_LANDMARKS)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.SELU()

    def forward(self, ldmks):
        # print(ldmks, ldmks.size())
        repr_ldmks = self.relu(self.dropout(self.fc1(ldmks)))

        self.prev_repr_ldmks.append(repr_ldmks)

        tensor_repr_ldmks = torch.cat(list(self.prev_repr_ldmks))
        if len(tensor_repr_ldmks.size()) <= 2:
            tensor_repr_ldmks = tensor_repr_ldmks.unsqueeze(0)

        out_rnn, (hidden, cells) = self.rnn(tensor_repr_ldmks)
        out_rnn = self.relu(out_rnn)
        out_linear = self.fc2(out_rnn)
        probas = self.softmax(out_linear)
        return probas
