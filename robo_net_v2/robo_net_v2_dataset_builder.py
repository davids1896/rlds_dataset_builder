from typing import Iterator, Tuple, Any
import os
import glob
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm
#import cv2

class RoboNetV2(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(640, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.'
                        ),
                        'depth': tfds.features.Image(
                            shape=(480, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Depth observation (single channel).'
                        ),
                        'depth_rgb': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Depth colorized RGB observation.'
                        ),
                        'depth_npy': tfds.features.Tensor(
                            shape=(640, 480),
                            dtype=np.float32,
                            doc='Raw depth data in numpy format.'
                        ),
                        'gripper_width': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Gripper width.'
                        ),
                        'pose': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='End-effector pose (position + orientation).'
                        ),
                        'joint': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint states of the robot arm.'
                        ),
                        'O_T_EE': tfds.features.Tensor(
                            shape=(4, 4),
                            dtype=np.float32,
                            doc='End-effector transformation matrix.'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action data.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                            doc='Language Instruction, e.g., "pick up the cube in the plate and put it in a bowl".'
                        )
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            })
        )
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/DavidHong/data/Franka/Franka_chenyang/new/Franka_Data/train/*'),
            'val': self._generate_examples(path='/home/DavidHong/data/Franka/Franka_chenyang/new/Franka_Data/val/*'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            data_json_path = os.path.join(episode_path, 'Franka/data.json')

            # load raw data --> this should change for your dataset
            with open(data_json_path, 'r') as f:
                    data = json.load(f)  # Use json.load() on the file object, not the path

           # data = json.load(data_json_path)     # this is a list of dicts in our case
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                pose = np.array([float(x) for x in step['pose'].split(',')]).astype(np.float32)
                joint = np.array([float(x) for x in step['joint'].split(',')]).astype(np.float32)
                O_T_EE = np.array([float(x) for x in step['O_T_EE'].split(',')]).reshape((4, 4)).astype(np.float32)
                # compute Kona language embedding
        #        language_embedding = self._embed([step['language_instruction']])[0].numpy()
         # 读取图像并解码
                def load_image_cv(filename, color_mode=cv2.IMREAD_COLOR):
                    image = cv2.imread(filename, color_mode)  # OpenCV reads images in BGR format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    return image
                
                image_path = os.path.join(episode_path, step['image'])
                depth_path = os.path.join(episode_path, step['depth'])
                depth_rgb_path = os.path.join(episode_path, step['depth_rgb'])
                depth_npy_path = os.path.join(episode_path, step['depth_npy'])



                # image_data = tf.io.read_file(f"{episode_path}/{step['image']}")
                # image = tf.image.decode_png(image_data, channels=3)
                # depth = tf.io.decode_png(tf.io.read_file(f"{episode_path}/{step['depth']}"), dtype=tf.uint16)
                # depth_rgb_data = tf.io.read_file(f"{episode_path}/{step['depth_rgb']}")
                # depth_rgb = tf.image.decode_png(depth_rgb_data, channels=3)
                image = load_image_cv(image_path)
                depth = load_image_cv(depth_path, color_mode=cv2.IMREAD_UNCHANGED)  # Depth image has single channel
                depth_rgb = load_image_cv(depth_rgb_path)


                depth_npy = np.load(depth_npy_path)

                #depth_data = tf.io.read_file(f"{episode_path}/{step['depth']}")
                #depth = tf.image.decode_png(depth_data, channels=1)
                
                #depth = depth.numpy()  # Convert Tensor to numpy array

                
                

                


                episode.append({
                    'observation': {
                        'image': image,
                        'depth': depth,
                        'depth_rgb': depth_rgb,
                        'depth_npy': depth_npy,
                        'gripper_width': float(step['gripper_width']),
                        'pose': pose,
                        'joint': joint,
                        'O_T_EE': O_T_EE
                    },
                    'action': np.zeros(7, dtype=np.float32),  # 示例: 可根据需要替换为实际 action 数据
                    #'discount': 1.0,
                    #'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    #'is_terminal': i == (len(data) - 1),
                    'language_instruction': 'pick up the cube in the plate and put it in a bowl',
                 #   'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in tqdm(episode_paths,desc="Generating examples", unit="episode"):
            
            yield _parse_example(sample)
            #input("stop here")

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

