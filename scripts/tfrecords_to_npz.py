import numpy as np
import tensorflow as tf


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
    for raw_record in raw_dataset:
        positions = np.array([])
        timesteps = []
        context, parsed_features = tf.io.parse_single_sequence_example(
            raw_record,
            context_features=context_features,
            sequence_features=sequence_features,
        )
        particle_types = context["particle_type"]
        particle_types = np.frombuffer(particle_types.values[0].numpy(), dtype=np.int64)

        for _, value in parsed_features.items():
            for i in range(len(value.values)):
                positions = np.frombuffer(value.values[i].numpy(), dtype=np.float32)

                # Load
                rows = len(positions) // 2
                positions = positions.reshape(rows, 2)
                timesteps.append(positions)

        simulation_data[f"simulation_{simulation_count}"] = (timesteps, particle_types)
        simulation_count += 1
    np.savez(output_path, simulation_data=simulation_data)


if __name__ == "__main__":
    convert_to_npz("original_data/WaterDrop/train.tfrecord", "processed_data/train.npz")
    convert_to_npz("original_data/WaterDrop/valid.tfrecord", "processed_data/valid.npz")
