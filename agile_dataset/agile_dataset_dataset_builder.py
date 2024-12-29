from typing import Iterator, Tuple, Any
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import io
import os
from tqdm import tqdm  # 引入 tqdm 库

class AgileDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for your custom dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
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
                        shape=(128,),  # The length of your action vector.
                        dtype=np.float32,
                        doc='Robot action, which consists of various control signals.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount factor for the step, default is 1.',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward for the step, typically 1 for the last step in the episode.',
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
                        doc='True if the episode has ended (terminal state).',
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
            'train': self._generate_examples(path='/mnt/hpfs/baaiei/realbench/Dataset/pick_up_the_banana_on_the_table_with_your_right_gripper_and_put_it_on_the_plate/train/episode_*.hdf5'),
            'val': self._generate_examples(path='/mnt/hpfs/baaiei/realbench/Dataset/pick_up_the_banana_on_the_table_with_your_right_gripper_and_put_it_on_the_plate/sample/episode_*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load the raw data (HDF5 format)
            with h5py.File(episode_path, 'r') as f:
                data = f['observations']  # Assuming observations are inside this group
                actions = f['action']  # Assuming actions are stored in this dataset
                qpos = f['observations/qpos']  # Correct path to qpos inside 'observations'

                episode = []
                num_steps = len(data['images']['cam_high'])  # Get number of steps (based on cam_high)

                for i in tqdm(range(num_steps), desc=f"Processing episode: {episode_path}", unit="step"):
                    # 获取图片的字节数据
                    cam_high_data = data['images']['cam_high'][i]
                    #cam_left_wrist_data = data['images']['cam_left_wrist'][i]
                    #cam_right_wrist_data = data['images']['cam_right_wrist'][i]

                    try:
                        # 使用 PIL 解码图像
                        cam_high = Image.open(io.BytesIO(cam_high_data))  # 将字节数据解码为图像
                        #cam_left_wrist = Image.open(io.BytesIO(cam_left_wrist_data))
                        #cam_right_wrist = Image.open(io.BytesIO(cam_right_wrist_data))

                        # 转换为 (640, 480) 并确保颜色模式为 RGB
                        cam_high = cam_high.convert('RGB').resize((224, 224))
                        #cam_left_wrist = cam_left_wrist.convert('RGB').resize((640, 480))
                        #cam_right_wrist = cam_right_wrist.convert('RGB').resize((640, 480))
                    except Exception as e:
                        print(f"Failed to load or process image at index {i}: {e}")
                        continue  # 跳过当前图像

                    # Create episode step
                    episode.append({
                        'observation': {
                            'cam_high': np.array(cam_high),  # Convert PIL image to np.array
                            #'cam_left_wrist': np.array(cam_left_wrist),
                            #'cam_right_wrist': np.array(cam_right_wrist),
                            'state': np.array(qpos[i]),  # Access qpos here
                        },
                        'action': np.array(actions[i]),
                        'discount': 1.0,
                        'reward': float(i == (num_steps - 1)),
                        'is_first': i == 0,
                        'is_last': i == (num_steps - 1),
                        'is_terminal': i == (num_steps - 1),
                        'language_instruction': 'pick up the banana on the table with your right gripper and put it on the plate.',
                    })

                # Create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path
                    }
                }

            # Return the parsed sample
            return episode_path, sample

        # Create list of all examples (episode files)
        episode_paths = glob.glob(path)

        # For smaller datasets, use single-thread parsing with tqdm for progress tracking
        for sample in tqdm(episode_paths, desc="Processing episodes", unit="file"):
            yield _parse_example(sample)
