from data_loader import DataLoader
from model import c3d_model as YasuoNet
from trainer import Trainer
from tensorflow.keras.optimizers import Adam
import argparse
import math

parser = argparse.ArgumentParser(description="Train YasuoNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--learning_rate", default=1e-3, type=float, dest='learning_rate')
parser.add_argument("--epochs", default=10, type=int, dest='epochs')
parser.add_argument("--batch_size", default=2, type=int, dest='batch_size')
parser.add_argument("--data_dir", default='./data', type=str, dest='data_dir')
parser.add_argument("--log_dir", default='./log', type=str, dest='log_dir')
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest='ckpt_dir')
# parser.add_argument("--result_dir", default='./results', type=str, dest='result_dir')

parser.add_argument("--mode", default='train', type=str, dest='mode')
# parser.add_argument("--train_continue", default='off', type=str, dest='train_continue')

args = parser.parse_args()

# parameter
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
data_dir = args.data_dir
log_dir = args.log_dir
ckpt_dir = args.ckpt_dir
# result_dir = args.result_dir
mode = args.mode
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

        model.compile(Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        trainer = Trainer(model, ckpt_dir, learning_rate, epochs)
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
