import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, x): return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
    def forward(self, x): return self.net(x)

class LinearPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.net(x)

def rk4_step(f, x, t, dt, steps=4):
    h = dt / steps
    for _ in range(steps):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + k1 * (h / 2))
        k3 = f(t + h / 2, x + k2 * (h / 2))
        k4 = f(t + h, x + k3 * h)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6.0)
        t = t + h
    return x

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
    def forward(self, t, x): return self.net(x)

class ODEPredictor(nn.Module):
    def __init__(self, rnn_hidden_dim, latent_dim):
        super().__init__()
        self.ode_func = ODEFunc(rnn_hidden_dim)
        self.projector = nn.Linear(rnn_hidden_dim, latent_dim)
    def forward(self, h_prev):
        h_next_cont = rk4_step(self.ode_func, h_prev, t=0.0, dt=1.0, steps=4)
        return self.projector(h_next_cont)

class CR_RPL_Passive_Model(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=128):
        super().__init__()
        self.enc_P = ConvEncoder(latent_dim=latent_dim)
        self.rnn = nn.GRUCell(latent_dim, rnn_hidden_dim)
        self.temporal_predictor_P = Predictor(rnn_hidden_dim, rnn_hidden_dim, latent_dim)
    def forward(self, g_P_t, h_prev):
        z_P = self.enc_P(g_P_t)
        z_P_hat_temp = self.temporal_predictor_P(h_prev)
        return z_P, z_P_hat_temp, self.rnn(z_P, h_prev)

class Linear_JEPA_Model(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=128):
        super().__init__()
        self.enc_P = ConvEncoder(latent_dim=latent_dim)
        self.rnn = nn.GRUCell(latent_dim, rnn_hidden_dim)
        self.temporal_predictor_P = LinearPredictor(rnn_hidden_dim, latent_dim)
    def forward(self, g_P_t, h_prev):
        z_P = self.enc_P(g_P_t)
        z_P_hat_temp = self.temporal_predictor_P(h_prev)
        return z_P, z_P_hat_temp, self.rnn(z_P, h_prev)

class NeuralODE_JEPA_Model(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=128):
        super().__init__()
        self.enc_P = ConvEncoder(latent_dim=latent_dim)
        self.rnn = nn.GRUCell(latent_dim, rnn_hidden_dim)
        self.temporal_predictor_P = ODEPredictor(rnn_hidden_dim, latent_dim)
    def forward(self, g_P_t, h_prev):
        z_P = self.enc_P(g_P_t)
        z_P_hat_temp = self.temporal_predictor_P(h_prev)
        return z_P, z_P_hat_temp, self.rnn(z_P, h_prev)

class RAE_Model(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=128):
        super().__init__()
        self.enc = ConvEncoder(latent_dim=latent_dim)
        self.rnn = nn.GRUCell(latent_dim, rnn_hidden_dim)
        self.dec = ConvDecoder(hidden_dim=rnn_hidden_dim)
    def forward(self, g_t, h_prev):
        h_t = self.rnn(self.enc(g_t), h_prev)
        return self.dec(h_t), h_t

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.GRUCell):
        nn.init.orthogonal_(m.weight_ih)
        nn.init.orthogonal_(m.weight_hh)

def variance_loss(z, gamma=1.0, eps=1e-4):
    return torch.mean(F.relu(gamma - torch.sqrt(z.var(dim=0) + eps)))

def compute_collapse_metrics(z):
    with torch.no_grad():
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t())
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(mask, 0)
        return sim_matrix.sum().item() / max(1, (sim_matrix.size(0) * (sim_matrix.size(0) - 1)))