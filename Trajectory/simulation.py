import torch
import math
import torch.nn.functional as F

def simulate_world_streams(batch_size, seq_len, canvas_size=64, device='cpu', movement_type='normal', shape_type='bar', force_vel_x=None, force_ang_vel=None, noise_std=0.0):
    template = torch.zeros((1, 1, canvas_size, canvas_size), device=device)
    center = canvas_size // 2

    if shape_type == 'bar':
        thickness = max(1, canvas_size // 16)
        length = canvas_size // 3
        template[:, :, center-length//2 : center+length//2, center-thickness//2 : center+thickness//2] = 1.0
    elif shape_type == 'circle':
        Y, X = torch.meshgrid(torch.arange(canvas_size, device=device), torch.arange(canvas_size, device=device), indexing='ij')
        dist = (X - center)**2 + (Y - center)**2
        radius = canvas_size // 6
        template[0, 0, dist <= radius**2] = 1.0
    elif shape_type == 'triangle':
        for i in range(canvas_size // 3):
            row = center - canvas_size//6 + i
            if 0 <= row < canvas_size:
                template[0, 0, row, max(0, center - i) : min(canvas_size, center + i + 1)] = 1.0

    world_seq = torch.zeros(batch_size, seq_len, 1, canvas_size, canvas_size, device=device)
    ground_truth = torch.zeros(batch_size, seq_len, 3, device=device) 

    for b in range(batch_size):
        pos_x, pos_y = 0.0, 0.0 
        
        vel_x = force_vel_x if force_vel_x is not None else (0.06 if torch.rand(1).item() > 0.5 else -0.06)
        angle = torch.rand(1).item() * 2 * math.pi
        angular_vel = force_ang_vel if force_ang_vel is not None else (0.25 if torch.rand(1).item() > 0.5 else -0.25)

        if movement_type == 'no_rotation':
            angular_vel = 0.0

        mid_point = seq_len // 2

        for t in range(seq_len):
            if t == mid_point:
                if movement_type == 'sudden_stop':
                    vel_x, angular_vel = 0.0, 0.0
                elif movement_type == 'teleport':
                    pos_x = -0.5 if pos_x > 0 else 0.5
                elif movement_type == 'reverse':
                    vel_x *= -1.0
                    angular_vel *= -1.0

            if movement_type == 'sine_wave':
                pos_y = math.sin(t * 0.5) * 0.5
            elif movement_type == 'accelerate':
                vel_x *= 1.1 

            if movement_type not in ['sudden_stop', 'accelerate'] and abs(pos_x + vel_x) > 0.7: 
                vel_x *= -1
                
            pos_x += vel_x
            angle += angular_vel
            
            ground_truth[b, t] = torch.tensor([pos_x, pos_y, angle])

            cos_a, sin_a = math.cos(angle), math.sin(angle)
            theta = torch.tensor([
                [cos_a, -sin_a, -pos_x],
                [sin_a,  cos_a, -pos_y]
            ], dtype=torch.float32, device=device).unsqueeze(0)

            grid = F.affine_grid(theta, template.size(), align_corners=False)
            world_seq[b, t] = F.grid_sample(template, grid, align_corners=False)[0]

    if noise_std > 0.0:
        noise = torch.randn_like(world_seq) * noise_std
        world_seq = torch.clamp(world_seq + noise, 0.0, 1.0)

    return world_seq, ground_truth