import json
import numpy as np
import torch
import torch_geometric as pyg


def get_metadata() -> dict:
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    return metadata


def load_npz(file_path: str) -> np.ndarray:
    with np.load(file_path, allow_pickle=True) as data_file:
        data = [item for _, item in data_file.items()]
    return data


def visualize_simulation(timesteps: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0.1, 0.9)
        timestep = timesteps[frame]
        ax.plot(timestep[:, 0], timestep[:, 1], "o", ms=2)

    ani = FuncAnimation(fig, update, frames=990)
    ani.save("simulation.mp4", writer="ffmpeg")


def generate_noise(position_seq: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps**0.5)
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat(
        (torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1
    )
    return position_noise


def to_graph(
    particle_type: torch.Tensor,
    position_seq: torch.Tensor,
    target_position: torch.Tensor,
    metadata: dict,
    noise_std: float = 0.0,
) -> pyg.data.Data:
    """Preprocess a trajectory and construct the graph"""
    # apply noise to the trajectory
    # position_noise = generate_noise(position_seq, noise_std)
    # position_seq = position_seq + position_noise

    # calculate the velocities of particles
    recent_position = position_seq[-1]
    velocity_seq = position_seq[0] - position_seq[-1]

    # construct the graph based on the distances between particles
    n_particle = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(
        recent_position,
        metadata["default_connectivity_radius"],
        loop=True,
        max_num_neighbors=n_particle,
    )

    # node-level features: velocity, distance to the boundary
    # normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat(
        (distance_to_lower_boundary, distance_to_upper_boundary), dim=-1
    )
    distance_to_boundary = torch.clip(
        distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0
    )

    # edge-level features: displacement, distance
    edge_displacement = torch.gather(
        recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, 2)
    ) - torch.gather(
        recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, 2)
    )
    edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # ground truth for training
    if target_position is not None:
        last_velocity = velocity_seq[-1]
        next_velocity = target_position - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(
            torch.tensor(metadata["acc_std"]) ** 2 + noise_std**2
        )
    else:
        acceleration = None

    # return the graph with features
    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration,
        pos=torch.cat((velocity_seq, distance_to_boundary), dim=1),
    )
    return graph


def rollout(
    model: torch.nn.Module,
    data: pyg.loader.DataLoader,
    metadata: dict,
    noise_std: float,
) -> torch.Tensor:
    model.eval()

    positions = torch.from_numpy(np.array(data.windows[0]["trajectories"]))
    particle_types = torch.from_numpy(np.array(data.windows[0]["particle_type"]))
    graph = data[0]

    window_size = 5
    num_timesteps = 1000

    traj = positions[: window_size - 1]

    for time in range(num_timesteps - window_size):
        with torch.no_grad():
            if time != 0:
                graph = to_graph(
                    particle_types, traj[-(window_size - 1) :], None, metadata
                )
            graph = graph.cuda()
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
