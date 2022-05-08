import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os

class Nim_Model(nn.Module):
    def __init__(self, action_size, hidden_size=128, num_lstm_layers=1, num_head_layers=1):
        super(Nim_Model, self).__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.lstm.flatten_parameters()

        self.policy_head = nn.Sequential(
            *[nn.Linear(in_features=hidden_size, out_features=hidden_size) for _ in range(num_head_layers-1)],
            nn.Linear(in_features=hidden_size, out_features=self.action_size)
        )
        self.value_head = nn.Sequential(
            *[nn.Linear(in_features=hidden_size, out_features=hidden_size) for _ in range(num_head_layers-1)],
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)  # add feature dimension
        
        h0 = x.new_zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        action_logits = self.policy_head(out)
        value_logit = self.value_head(out)

        return F.softmax(action_logits, dim=-1), torch.tanh(value_logit)

    def predict(self, state):
        if len(state.shape) != 1:
            raise Exception('predict function only processes individual state')
        
        device = next(self.lstm.parameters()).device
        state = torch.FloatTensor(state.astype(np.float32)).to(device)
        state = torch.unsqueeze(state, dim=0)
        
        with torch.no_grad():
            policy, value = self(state)
        return policy.squeeze().data.cpu().numpy(), value.item()

    def save_checkpoint(self, folder='.', filename='checkpoint_model'):
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, filename)
        torch.save(self.state_dict(), filepath)
        
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)



if __name__ == '__main__':
    from NimEnvironments import NimEnv

    game = NimEnv(num_piles=6)
    # model = Nim_Model(game.board_size, game.action_size, device='cpu')
    model = Nim_Model(action_size=game.action_size, hidden_size=128, num_lstm_layers=1, num_head_layers=1).to('cpu')
    pred = model(torch.FloatTensor([[1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, 1, -1, 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  1,  1,  1,  1,  1,1,  1,  1,  1,  1], [1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, 1, -1, 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  1,  1,  1,  1,  1,1,  1,  1,  1,  1], [1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, 1, -1, 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  1,  1,  1,  1,  1,1,  1,  1,  1,  1]]))
    print(pred)
