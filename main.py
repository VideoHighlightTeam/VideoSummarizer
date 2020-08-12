from data_loader import DataLoader
from model import c3d_model as YasuoNet
from trainer import Trainer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils import class_weight
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description="Train YasuoNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default='train', type=str, dest='mode', required=True)
parser.add_argument("--data_dir", type=str, dest='data_dir', required=True)
parser.add_argument("--batch_size", type=int, dest='batch_size', required=True)
parser.add_argument("--epochs", type=int, dest='epochs', required=True)
parser.add_argument("--learning_rate", default=1e-3, type=float, dest='learning_rate')
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest='ckpt_dir')
# parser.add_argument("--train_continue", default='off', type=str, dest='train_continue')

args = parser.parse_args()

# parameter
mode = args.mode
data_dir = args.data_dir
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
ckpt_dir = args.ckpt_dir
# train_continue = args.train_continue


def main():
    data_loader = DataLoader(data_dir)
    data_config = data_loader.get_metadata()['config']
    video_frames = int(data_config['segment_length'] * data_config['video_sample_rate'])
    video_width = data_config['video_width']
    video_height = data_config['video_height']
    video_channels = 3  # 향후 메타데이터에서 읽어오도록 수정

    model = YasuoNet(video_frames, video_width, video_height, video_channels)

    if mode == 'train':
        # epoch 당 배치 수
        train_steps = data_loader.get_train_data_count() // batch_size
        valid_steps = math.ceil(data_loader.get_valid_data_count() / batch_size)

        model.compile(Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

        class_weights = class_weight.compute_class_weight('balanced', classes=range(data_loader.CLASS_COUNT), y=data_loader.all_segment_df['label'].to_numpy())
        class_weights = {i: v for i, v in enumerate(class_weights)}

        trainer = Trainer(model, ckpt_dir, learning_rate, epochs, class_weights)
        trainer.train(
            data_loader.iter_train_batch_data(batch_size), train_steps,
            data_loader.iter_valid_batch_data(batch_size), valid_steps
        )
    elif mode == 'test':
        test_steps = math.ceil(data_loader.get_test_data_count() / batch_size)

        trainer = Trainer(model, ckpt_dir, learning_rate, epochs)
        trainer.test(data_loader.iter_test_batch_data(batch_size), test_steps)
    elif mode == 'predict':
        pass


if __name__ == '__main__':
    main()
