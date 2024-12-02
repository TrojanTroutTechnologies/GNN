import json
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def get_metadata() -> dict:
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    return metadata


def load_npz(file_path: str) -> np.ndarray:
    with np.load(file_path, allow_pickle=True) as data_file:
        data = [item for _, item in data_file.items()]
    return data


def visualize_simulation(timesteps: np.ndarray, checkpoint: int) -> None:
    fig, ax = plt.subplots()
    progress = tqdm(total=len(timesteps), desc="Creating Animation")

    def update(frame):
        progress.update(1)
        ax.clear()
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0.1, 0.9)
        timestep = timesteps[frame]
        ax.plot(timestep[:, 0], timestep[:, 1], "o", ms=2)

    ani = FuncAnimation(
        fig, update, frames=np.arange(0, len(timesteps)), interval=10, blit=False
    )

    ani.save("simulation.mp4", writer="ffmpeg")

    progress.close()
    plt.close(fig)


def generate_noise(
    position_seq: torch.Tensor, velocity_seq: torch.Tensor, noise_std: float
) -> torch.Tensor:
    """Generate noise for a trajectory"""
    # Generate Gaussian noise
    noise = torch.randn_like(velocity_seq) * (noise_std / velocity_seq.size(0) ** 0.5)
    # Accumulate noise as a random walk
    accumulated_noise = noise.cumsum(dim=0)

    # Perturb the velocity sequence with the accumulated noise
    noisy_velocity_seq = velocity_seq + accumulated_noise

    # Adjust positions to maintain consistency with the noisy velocities
    noisy_position_seq = position_seq.clone()
    for i in range(1, position_seq.size(0)):
        noisy_position_seq[i] = noisy_position_seq[i - 1] + noisy_velocity_seq[i - 1]

    return noisy_position_seq, noisy_velocity_seq


def to_graph(
    particle_type: torch.Tensor,
    position_seq: torch.Tensor,
    target_position: torch.Tensor,
    metadata: dict,
    noise_std: float = 0.0,
) -> pyg.data.Data:
    """Preprocess a trajectory and construct the graph"""
    # Compute velocity sequence and add noise
    velocity_seq = position_seq[1:] - position_seq[:-1]
    position_seq, velocity_seq = generate_noise(position_seq, velocity_seq, noise_std)
    recent_position = position_seq[-1]

    # Construct the graph using radius-based connectivity
    num_particles = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(
        recent_position,
        metadata["default_connectivity_radius"],
        loop=True,
        max_num_neighbors=num_particles - 1,
    )

    # Node-level features: Normalize velocity
    vel_mean = torch.tensor(metadata["vel_mean"]).view(
        1, -1
    )  # Match shape of velocity_seq
    vel_std = torch.tensor(metadata["vel_std"]).view(1, -1)
    normal_velocity_seq = (velocity_seq - vel_mean) / torch.sqrt(
        vel_std**2 + noise_std**2
    )

    # Compute distances to the boundary
    boundary = torch.tensor(metadata["bounds"])  # Shape: [2, 2]
    boundary = boundary.unsqueeze(0).expand(
        recent_position.size(0), -1, -1
    )  # Match recent_position
    distance_to_lower_boundary = recent_position - boundary[:, :, 0]
    distance_to_upper_boundary = boundary[:, :, 1] - recent_position
    distance_to_boundary = torch.cat(
        (distance_to_lower_boundary, distance_to_upper_boundary), dim=-1
    )

    # use tanh to normalize the distance to [-1, 1]
    distance_to_boundary = torch.tanh(distance_to_boundary)

    # Edge-level features: Displacement and distance
    edge_displacement = recent_position[edge_index[0]] - recent_position[edge_index[1]]
    edge_displacement /= metadata["default_connectivity_radius"] + 1e-8
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # Compute ground truth acceleration for training (if target_position is provided)
    if target_position is not None:
        last_velocity = velocity_seq[-1]
        next_velocity = target_position - recent_position
        acceleration = next_velocity - last_velocity
        acc_mean = torch.tensor(metadata["acc_mean"]).view(
            1, -1
        )  # Match acceleration shape
        acc_std = torch.tensor(metadata["acc_std"]).view(1, -1)
        acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std**2 + noise_std**2)
    else:
        acceleration = torch.zeros_like(
            recent_position
        )  # Default to zero for inference

    # Construct and return the graph
    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration,
        pos=torch.cat(
            (
                normal_velocity_seq.reshape(normal_velocity_seq.size(1), -1),
                distance_to_boundary,
            ),
            dim=-1,
        ),
    )
    return graph


def rollout(
    model: torch.nn.Module,
    data: pyg.loader.DataLoader,
    metadata: dict,
    noise_std: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    model.eval()

    positions = torch.from_numpy(np.array(data.windows[0]["trajectories"]))
    particle_types = torch.from_numpy(np.array(data.windows[0]["particle_type"]))
    graph = data[0]

    window_size = model.window_size
    num_timesteps = 1000

    traj = positions[: window_size - 1]

    print("Rolling out the model...")
    for time in range(num_timesteps - window_size):
        with torch.no_grad():
            if time != 0:
                graph = to_graph(
                    particle_types, traj[-(window_size - 1) :], None, metadata, 0.0
                )
            graph = graph.to(device)
            acceleration = model(graph).cpu()

            acceleration = acceleration * torch.sqrt(
                torch.tensor(metadata["acc_std"]) ** 2 + noise_std**2
            ) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[-1]

            recent_velocity = recent_position - traj[-2]

            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity

            # update traj
            traj = torch.cat((traj, new_position.unsqueeze(0)), dim=0)

    return traj
