import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Iterator, Tuple, Any
import glob
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import io
import os
from tqdm import tqdm

# Helper functions to process poses (same as you provided)
def convert_quaternion_to_euler(quat):
    """Convert Quaternion (xyzw) to Euler angles (rpy)."""
    quat = quat / np.linalg.norm(quat)
    euler = R.from_quat(quat).as_euler('xyz')
    return euler

def convert_euler_to_quaternion(euler):
    """Convert Euler angles (rpy) to Quaternion (xyzw)."""
    quat = R.from_euler('xyz', euler).as_quat()
    return quat

def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = np.stack((i, j, k), axis=1)
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix

def compute_ortho6d_from_rotation_matrix(matrix):
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d

# New helper function: Convert rotation matrix to Euler angles
def convert_rotation_matrix_to_euler(rotation_matrix):
    """Convert a rotation matrix to Euler angles."""
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    return euler_angles

# Updated Dataset class
class AgileDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for custom Agile dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (features, splits,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'cam_high': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(128,),  # Assuming state has 128 elements.
                            dtype=np.float32,
                            doc='Robot state (e.g., joint positions, gripper positions, etc.).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),  # Modified action vector size
                        dtype=np.float32,
                        doc='Modified action vector: [gripper, eef_position, 3D pose]',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount factor for the step.',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward for the step.',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if this is the first step of the episode.',
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if this is the last step of the episode.',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the episode has ended.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/realbench/Dataset/pick_up_the_banana_on_the_table_with_your_right_gripper_and_put_it_on_the_plate/train/episode_*.hdf5'),
            'val': self._generate_examples(path='/home/realbench/Dataset/pick_up_the_banana_on_the_table_with_your_right_gripper_and_put_it_on_the_plate/sample/episode_*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load raw data (HDF5 format)
            with h5py.File(episode_path, 'r') as f:
                data = f['observations']  # Assuming observations are inside this group
                actions = f['action']  # Assuming actions are stored here
                qpos = f['observations/qpos']  # Correct path to qpos inside 'observations'

                episode = []
                num_steps = len(data['images']['cam_high'])  # Number of steps

                for i in tqdm(range(num_steps), desc=f"Processing episode: {episode_path}", unit="step"):
                    cam_high_data = data['images']['cam_high'][i]

                    try:
                        cam_high = Image.open(io.BytesIO(cam_high_data))
                        cam_high = cam_high.convert('RGB').resize((224, 224))
                    except Exception as e:
                        print(f"Failed to load image at index {i}: {e}")
                        continue

                    # Extract the necessary data for action
                    gripper_position = qpos[i][10]  # Assuming gripper position is at index 10
                    eef_position = qpos[i][30:32]  # Assuming EEF position is at indices 30~32
                    pose_6d = qpos[i][33:39]  # 6D pose (33~38) -> Quaternion and rotation

                    # Convert 6D pose to 3D position
                    rotation_matrix = compute_rotation_matrix_from_ortho6d(pose_6d[None, :])  # (1, 3, 3)
                    euler_angles = convert_rotation_matrix_to_euler(rotation_matrix[0])  # Extract Euler angles

                    gripper_position = np.array([gripper_position])  # 转为一维数组
                    eef_position = np.array(eef_position)  # 保证是数组类型
                    euler_angles = np.array(euler_angles[:3])  # 提取前三个欧拉角并转为数组

                    action = np.concatenate([gripper_position, eef_position, euler_angles[:3]])  # Combine for 7D action

                    # Create episode step
                    episode.append({
                        'observation': {
                            'cam_high': np.array(cam_high),
                            'state': np.array(qpos[i]),
                        },
                        'action': action,
                        'discount': 1.0,
                        'reward': float(i == (num_steps - 1)),
                        'is_first': i == 0,
                        'is_last': i == (num_steps - 1),
                        'is_terminal': i == (num_steps - 1),
                        'language_instruction': 'pick up the banana on the table with your right gripper and put it on the plate.',
                    })

                # Return parsed sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {'file_path': episode_path}
                }

            return episode_path, sample

        episode_paths = glob.glob(path)
        for sample in tqdm(episode_paths, desc="Generating examples", unit="episode"):
            yield _parse_example(sample)
