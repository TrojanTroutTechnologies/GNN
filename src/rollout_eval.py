from utils import visualize_simulation, rollout, get_metadata
from gnn_network import LearnedSimulator
from train import OneStepDataset
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="1000")
    args = parser.parse_args()

    # Load the model
    checkpoint = torch.load(
        f"checkpoints/checkpoint_{args.checkpoint}.pt", weights_only=True
    )
    model = LearnedSimulator(window_size=7)
    model = model.cuda()
    model.load_state_dict(checkpoint["model"])

    metadata = get_metadata()

    # Create a dataset
    dataset = OneStepDataset(
        "data/processed/valid.npz", metadata, window_size=7, noise_std=0.0
    )

    # Rollout the model
    traj = rollout(model, dataset, metadata, 0.0)

    # Visualize the simulation
    visualize_simulation(traj, checkpoint)
