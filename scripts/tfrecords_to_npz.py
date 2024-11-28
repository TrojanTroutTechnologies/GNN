import numpy as np
import tensorflow as tf
from tqdm import tqdm


def convert_to_npz(file_path, output_path):
    context_features = {
        "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "particle_type": tf.io.VarLenFeature(tf.string),
    }

    sequence_features = {
        "position": tf.io.VarLenFeature(tf.string),
    }

    simulation_count = 0
    simulation_data = {}

    raw_dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in tqdm(raw_dataset):
        timesteps = []
        try:
            context, parsed_features = tf.io.parse_single_sequence_example(
                raw_record,
                context_features=context_features,
                sequence_features=sequence_features,
            )
            if "particle_type" not in context or not context["particle_type"].values:
                continue
            particle_types = context["particle_type"]
            particle_types = np.frombuffer(
                particle_types.values[0].numpy(), dtype=np.int64
            )

            for feature, value in parsed_features.items():
                if not value.values:
                    continue
                for i in range(len(value.values)):
                    positions = np.frombuffer(value.values[i].numpy(), dtype=np.float32)
                    if len(positions) % 2 != 0:
                        print(f"Skipping malformed position data in feature {feature}")
                        continue
                    rows = len(positions) // 2
                    positions = positions.reshape(rows, 2)
                    timesteps.append(positions)

            simulation_data[f"simulation_{simulation_count}"] = (
                timesteps,
                particle_types,
            )
            simulation_count += 1
        except Exception as e:
            print(f"Error processing record: {e}")

    np.savez(output_path, simulation_data=simulation_data)


if __name__ == "__main__":
    convert_to_npz("data/original/WaterDrop/train.tfrecord", "data/processed/train.npz")
    convert_to_npz("data/original/WaterDrop/valid.tfrecord", "data/processed/valid.npz")
