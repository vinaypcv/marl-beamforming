import torch
import torch.nn as nn

class WirelessActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(WirelessActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.feature_dim = state_dim // 8
        
        self.temporal_processor = nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        
        self.fc_policy = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        batch_size = state.size(0) if state.dim() > 1 else 1
        x = state.view(batch_size, 8, -1)
        gru_out, _ = self.temporal_processor(x)
        last_time_step = gru_out[:, -1, :]
        return self.fc_policy(last_time_step)

class CentralizedWirelessCritic(nn.Module):
    def __init__(self, state_dim_ue, state_dim_gnb, action_dim_ue, action_dim_gnb):
        super(CentralizedWirelessCritic, self).__init__()
        total_input_dim = state_dim_ue + state_dim_gnb + action_dim_ue + action_dim_gnb
        self.maddpg_critic = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s_ue, s_gnb, a_ue, a_gnb):
        inputs = torch.cat([s_ue, s_gnb, a_ue, a_gnb], dim=-1)
        return self.maddpg_critic(inputs)
