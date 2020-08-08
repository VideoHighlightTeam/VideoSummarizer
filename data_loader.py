import os
import glob
import numpy as np
import pandas as pd
import hashlib
from functools import reduce

import util.collection_util as cu


class DataLoader:
    """ Data Converter가 생성한 segment 파일(.pkl)을 학습/검증/테스트 절차에 맞게 배치 단위로 공급하는 기능 구현 """

    CLASS_COUNT = 2

    def __init__(self, dataset_dir, train_prop=0.6, valid_prop=0.2):
        self.dataset_dir = dataset_dir
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.test_prop = 1 - train_prop - valid_prop

        self.all_segment_df = self._get_all_segment_df()

        self.train_segment_df, self.valid_segment_df, self.test_segment_df = self._split_dataset()

    def _get_all_segment_df(self):
        all_segment_path_list = sorted(glob.glob(self.dataset_dir + '/*/*.pkl'))

        # segment 정보를 1차원 리스트로 나열
        all_segment_list = []
        for path in all_segment_path_list:
            # dir: 원본영상 이름, name: 파일 이름
            title, name = os.path.normpath(path).split(os.sep)[-2:]
            name = os.path.splitext(name)[0]
            # 원본영상 내의 segment index
            index = int(name.split('_')[1])
            # 이 segment의 label
            label = int(os.path.splitext(name)[0][-1])

            all_segment_list.append({'title': title, 'name': name, 'index': index, 'label': label, 'path': path})

        return pd.DataFrame(all_segment_list)

    def _split_dataset(self):
        def get_subset(title):
            hashing = hashlib.sha512()
            hashing.update(title.encode())
            digest = hashing.digest()

            hvalue = reduce(lambda x, y: x ^ y, digest) / 255

            if hvalue < self.train_prop:
                return 'train'
            elif hvalue < self.train_prop + self.valid_prop:
                return 'valid'
            else:
                return 'test'

        df = self.all_segment_df
        df['subset'] = df['title'].apply(get_subset)

        return df[df['subset'] == 'train'], df[df['subset'] == 'valid'], df[df['subset'] == 'test']

    def get_train_data_count(self):
        return len(self.train_segment_df)

    def get_valid_data_count(self):
        return len(self.valid_segment_df)

    def get_test_data_count(self):
        return len(self.test_segment_df)

    def get_all_data_count(self):
        return len(self.all_segment_df)

    def get_data_count_by_label(self, label):
        return sum(self.all_segment_df['label'] == label)

    def iter_train_batch_data(self, batch_size, repeat=True):
        """ 학습 데이터에서 batch_size만큼씩 중복없이 무작위 샘플하여 순차적으로 반환 """

        # 전체 데이터를 순회할 때까지 반복
        while True:
            segment_label_df = [self.train_segment_df[self.train_segment_df['label'] == c].sample(frac=1) for c in range(self.CLASS_COUNT)]

            # 각 label별로 배치 데이터에 포함될 개수
            label_size = np.zeros(self.CLASS_COUNT, dtype=np.int)
            for c in range(self.CLASS_COUNT - 1):
                label_size[c] = int(len(segment_label_df[c]) / len(self.train_segment_df) * batch_size)
            label_size[-1] = batch_size - label_size.sum()

            # 전체 학습 데이터에서 현재 iterator의 위치
            i = np.zeros(self.CLASS_COUNT, dtype=np.int)

            while True:
                # 각 label별 배치 데이터 slicing
                label_batch_df = [segment_label_df[c].iloc[i[c]: i[c] + label_size[c]] for c in range(self.CLASS_COUNT)]

                # 레이블별로 가져온 데이터 수의 합이 batch_size보다 작으면 iteration 종료
                if sum(map(len, label_batch_df)) < batch_size:
                    break

                # 모든 레이블 배치 데이터에 대해 segment 데이터 읽어와서 리스트 생성
                batch_data = []
                for c in range(self.CLASS_COUNT):
                    batch_data += [cu.load(segment['path']) for _, segment in label_batch_df[c].iterrows()]

                # x, y 데이터 분리
                # batch_data_x, batch_data_y = zip(*[((segment['video'], segment['audio']), segment['label']) for segment in batch_data])
                batch_data_x, batch_data_y = zip(*[(segment['video'], segment['label']) for segment in batch_data])

                batch_data_x = np.array(batch_data_x)
                batch_data_y = np.array(batch_data_y).reshape(-1, 1)

                # 데이터를 iterator로 반환
                yield batch_data_x, batch_data_y

                i += label_size

            if not repeat:
                break


    def iter_valid_batch_data(self, batch_size, repeat=True):
        """ 검증 데이터에서 batch_size만큼씩 순차적으로 반환 """
        for batch_data in self._iter_subset_batch_data(self.valid_segment_df, batch_size, repeat):
            yield batch_data

    def iter_test_batch_data(self, batch_size, repeat=True):
        """ 테스트 데이터에서 batch_size만큼씩 순차적으로 반환 """
        for batch_data in self._iter_subset_batch_data(self.test_segment_df, batch_size, repeat):
            yield batch_data

    def iter_all_batch_data(self, batch_size, repeat=True):
        """ 전체 데이터에서 batch_size만큼씩 순차적으로 반환 """
        for batch_data in self._iter_subset_batch_data(self.all_segment_df, batch_size, repeat):
            yield batch_data

    def _iter_subset_batch_data(self, subset_df, batch_size, repeat=True):
        # 전체 데이터를 순회할 때까지 반복
        while True:
            # 주어진 데이터에서 현재 iterator의 위치
            i = 0

            while True:
                # 배치 데이터 slicing
                batch_df = subset_df.iloc[i: i + batch_size]

                # 데이터가 없으면 iteration 종료
                if len(batch_df) == 0:
                    break

                # 모든 배치 데이터에 대해 segment 데이터 읽어와서 리스트 생성
                batch_data = [cu.load(segment['path']) for _, segment in batch_df.iterrows()]

                # x, y 데이터 분리
                # batch_data_x, batch_data_y = zip(*[((segment['video'], segment['audio']), segment['label']) for segment in batch_data])
                batch_data_x, batch_data_y = zip(*[(segment['video'], segment['label']) for segment in batch_data])

                batch_data_x = np.array(batch_data_x)
                batch_data_y = np.array(batch_data_y).reshape(-1, 1)

                # 데이터를 iterator로 반환
                yield batch_data_x, batch_data_y

                i += batch_size

            if not repeat:
                break
