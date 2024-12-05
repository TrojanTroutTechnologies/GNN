from src.gnn import LearnedSimulator
from src.train import OneStepDataset
from src.utils import visualize_simulation, rollout, get_metadata

import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load the model
    checkpoint = torch.load(
        f"checkpoints/epoch_{args.epoch}/checkpoint_{args.checkpoint}.pt",
        weights_only=True,
    )
    model = LearnedSimulator(window_size=args.window_size)
    model = model.to(args.device)
    model.load_state_dict(checkpoint["model"])

    metadata = get_metadata()

    # Create a dataset
    dataset = OneStepDataset(
        "data/processed/valid.npz",
        metadata,
        window_size=args.window_size,
        noise_std=0.0,
    )

    # Rollout the model
    traj = rollout(model, dataset, metadata, args.device)

    # Visualize the simulation
    visualize_simulation(traj, checkpoint)
