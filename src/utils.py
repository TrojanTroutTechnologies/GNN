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


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, path, metadata, window_size=5, noise_std=0.0):
        super().__init__()
        self.data = load_npz(path)
        self.metadata = metadata
        self.noise_std = noise_std

        self.window_size = window_size
        self.windows = []
        self._create_windows()

    def _create_windows(self):
        for sim in self.data[0].tolist().values():
            timesteps = sim[0]
            for i in range(0, len(timesteps) - self.window_size):
                trajectories = timesteps[i : i + self.window_size]
                assert len(trajectories) == self.window_size
                window = {
                    "size": len(sim[1]),
                    "particle_type": sim[1],
                    "trajectories": trajectories,
                }
                self.windows.append(window)

    def len(self):
        return len(self.windows)

    def get(self, idx):
        window = self.windows[idx]
        particle_type = torch.from_numpy(window["particle_type"])
        trajectories = window["trajectories"]
        target_position = np.array(trajectories[-1])
        position_seq = np.array(trajectories[:-1])
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)

        with torch.no_grad():
            graph = to_graph(
                particle_type,
                position_seq,
                target_position,
                self.metadata,
                self.noise_std,
            )
        return graph


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
) -> (torch.Tensor, torch.Tensor):
    """Generate noise for a trajectory."""
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


def compute_normalized_velocity(
    velocity_seq: torch.Tensor, metadata: dict, noise_std: float
) -> torch.Tensor:
    """Compute normalized velocity given velocity sequence and metadata."""
    vel_mean = torch.tensor(metadata["vel_mean"]).view(1, -1)
    vel_std = torch.tensor(metadata["vel_std"]).view(1, -1)
    normal_velocity_seq = (velocity_seq - vel_mean) / torch.sqrt(
        vel_std**2 + noise_std**2
    )
    return normal_velocity_seq


def compute_distance_to_boundary(
    recent_position: torch.Tensor, metadata: dict
) -> torch.Tensor:
    """Compute normalized distance to boundary for each particle."""
    boundary = torch.tensor(metadata["bounds"])
    boundary = boundary.unsqueeze(0).expand(recent_position.size(0), -1, -1)
    distance_to_lower_boundary = recent_position - boundary[:, :, 0]
    distance_to_upper_boundary = boundary[:, :, 1] - recent_position
    distance_to_boundary = torch.cat(
        (distance_to_lower_boundary, distance_to_upper_boundary), dim=-1
    )
    # Normalize distances to [-1, 1]
    distance_to_boundary = torch.tanh(
        distance_to_boundary / metadata["default_connectivity_radius"]
    )
    return distance_to_boundary


def compute_edges(
    recent_position: torch.Tensor, metadata: dict
) -> (torch.Tensor, torch.Tensor):
    """Compute the graph edges and their attributes (displacements, distances)."""
    num_particles = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(
        recent_position,
        metadata["default_connectivity_radius"],
        loop=True,
        max_num_neighbors=num_particles - 1,
    )
    edge_displacement = recent_position[edge_index[0]] - recent_position[edge_index[1]]
    edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
    return edge_index, torch.cat((edge_displacement, edge_distance), dim=-1)


def compute_acceleration(
    recent_position: torch.Tensor,
    velocity_seq: torch.Tensor,
    target_position: torch.Tensor,
    metadata: dict,
    noise_std: float,
) -> torch.Tensor:
    """Compute normalized acceleration if target_position is given, else return zeros."""
    if target_position is not None:
        last_velocity = velocity_seq[-1]
        next_velocity = target_position - recent_position
        acceleration = next_velocity - last_velocity
        acc_mean = torch.tensor(metadata["acc_mean"]).view(1, -1)
        acc_std = torch.tensor(metadata["acc_std"]).view(1, -1)
        acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std**2 + noise_std**2)
    else:
        acceleration = torch.zeros_like(recent_position)
    return acceleration


def to_graph(
    particle_type: torch.Tensor,
    position_seq: torch.Tensor,
    target_position: torch.Tensor,
    metadata: dict,
    noise_std: float = 0.0,
) -> pyg.data.Data:
    """Preprocess a trajectory and construct the graph."""

    velocity_seq = position_seq[1:] - position_seq[:-1]
    position_seq, velocity_seq = generate_noise(position_seq, velocity_seq, noise_std)
    recent_position = position_seq[-1]

    edge_index, edge_attr = compute_edges(recent_position, metadata)

    normal_velocity_seq = compute_normalized_velocity(velocity_seq, metadata, noise_std)

    distance_to_boundary = compute_distance_to_boundary(recent_position, metadata)

    # Compute acceleration (if target_position is provided)
    acceleration = compute_acceleration(
        recent_position, velocity_seq, target_position, metadata, noise_std
    )

    final_node_features = torch.cat(
        (
            normal_velocity_seq.reshape(normal_velocity_seq.size(1), -1),
            distance_to_boundary,
        ),
        dim=-1,
    )

    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=acceleration,
        pos=final_node_features,
    )
    return graph


def rollout(
    model: torch.nn.Module,
    data: pyg.loader.DataLoader,
    metadata: dict,
    device: str = "cuda",
) -> torch.Tensor:
    model.eval()

    positions = torch.from_numpy(np.array(data.windows[0]["trajectories"]))
    particle_types = torch.from_numpy(np.array(data.windows[0]["particle_type"]))
    graph = data[0]

    window_size = model.window_size
    num_timesteps = len(data.data[0].tolist()["simulation_0"][0])

    traj = positions[: window_size - 1]

    print("Rolling out the model...")
    for time in range(num_timesteps - window_size + 1):
        with torch.no_grad():
            if time != 0:
                graph = to_graph(
                    particle_types, traj[-(window_size - 1) :], None, metadata, 0.0
                )
            graph = graph.to(device)
            acceleration = model(graph).cpu()

            acceleration = acceleration * torch.sqrt(
                torch.tensor(metadata["acc_std"]) ** 2
            ) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[-1]

            recent_velocity = recent_position - traj[-2]

            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity

            # update traj
            traj = torch.cat((traj, new_position.unsqueeze(0)), dim=0)

    return traj
