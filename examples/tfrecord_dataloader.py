import os 
import sys, time
import tensorflow as tf 
import torch 
from torch.utils.data import IterableDataset, DataLoader


version = sys.argv[1] if len(sys.argv) > 1 else "v1001"
path = sys.argv[2] if len(sys.argv) > 2 else "202506"

base_local_path = path
print("Base path:", base_local_path)
files = []


def get_timestamp():
    t = time.time()
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{s}.{ms:03d}"

def get_file_list():
    """获取 Parquet 文件列表"""
    files = []
    if os.path.exists(base_local_path):
        for d in os.listdir(base_local_path):
            if d.endswith(".tfrecord.gz"):
                files.append(os.path.join(base_local_path, d))
    files.sort()
    print(f"Found {len(files)} Parquet files.")
    return files

class StreamingTFRecordDataset(IterableDataset):
    def __init__(self, tfrecord_files, worker_id, total_workers, batch_size=512):
        """
        流式读取TFRecord数据集
        :param tfrecord_files: TFRecord文件路径列表
        :param worker_id: 当前工作进程ID
        :param total_workers: 总工作进程数
        :param batch_size: 批大小
        """
        self.tfrecord_files = tfrecord_files
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.batch_size = batch_size
        
        # 定义特征规格
        self.seq_features_config = {
            'dis_51':[0, 5], 'dis_52':[0, 5], 'dis_53':[0, 5], 'dis_54':[0, 5], 
            'dis_55':[0, 30], 'dis_56':[0, 30],
            
            'dis_60':[0, 30],
            'dis_61':[0, 30], 
        }

    def _parse_tfrecord(self, example_proto):
        """解析单个TFRecord样本"""
        features = {
            'play_click': tf.io.FixedLenFeature([], tf.int64),
            'play_score': tf.io.FixedLenFeature([], tf.float32),
            'play_1m_label': tf.io.FixedLenFeature([], tf.int64),
            'play_5m_label': tf.io.FixedLenFeature([], tf.int64),
            'play_9m_label': tf.io.FixedLenFeature([], tf.int64),
        }

        # 添加离散特征 (dis_00 to dis_43)
        for i in range(51):
            features[f'dis_{i:02d}'] = tf.io.FixedLenFeature([], tf.int64)
            
        # 添加序列特征
        for key in self.seq_features_config.keys():
            features[key] = tf.io.VarLenFeature(tf.int64)
            
        parsed_features = tf.io.parse_example(example_proto, features)
        
        # 处理序列特征
        for key, expected_length in self.seq_features_config.items():
            # 将稀疏张量转为稠密张量
            dense_tensor = tf.sparse.to_dense(parsed_features[key], default_value=0)
            # 获取当前长度并填充到期望长度
            current_length = tf.shape(dense_tensor)[0]
            padding_length = expected_length[1] - current_length
            # 填充序列
            # paddings = tf.stack([[0, padding_length]], axis=0)
            padded_tensor = tf.pad(dense_tensor, [[0, padding_length]], constant_values=0)
            parsed_features[key] = padded_tensor
            
        return parsed_features
    
    def _create_dataset(self):
        """创建流式tf.data.Dataset"""
        # 创建文件列表
        files_ds = tf.data.Dataset.from_tensor_slices(self.tfrecord_files)
        
        # 根据worker_id进行数据划分
        # files_ds = files_ds.shard(self.total_workers, self.worker_id)
        
        # 读取TFRecord文件
        dataset = files_ds.interleave(
            lambda file_path: tf.data.TFRecordDataset(file_path, compression_type='GZIP', buffer_size=1024*1024),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 解析记录
        
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __iter__(self):
        """迭代器实现流式读取"""
        dataset = self._create_dataset()
        
        # 使用tf.data迭代数据
        for batch in dataset.as_numpy_iterator():
            # 提取特征和标签
            discrete_features = {}
            for i in range(44):  # dis_00 to dis_43
                discrete_features[f'dis_{i:02d}'] = torch.tensor(batch[f'dis_{i:02d}'], dtype=torch.long)
            
            # seq_features1 = {}
            # seq_features2 = {}
            # seq_keys1 = ['dis_44', 'dis_45', 'dis_46', 'dis_47']
            # seq_keys2 = ['dis_48', 'dis_49', 'dis_50', 'dis_51', 'dis_52', 'dis_53']
            seq_features = {}
            seq_keys = ['dis_51', 'dis_52', 'dis_53', 'dis_54', 'dis_55', 'dis_56',  'dis_60', 'dis_61',]
            for key in seq_keys:
                seq_features[key] = torch.tensor(batch[key], dtype=torch.long)
            
            # for key in seq_keys1:
            #     seq_features1[key] = torch.tensor(batch[key], dtype=torch.long)
            #     # seq_features1.append(batch[key])
            
            # for key in seq_keys2:
            #     seq_features2[key] = torch.tensor(batch[key], dtype=torch.long)

            # 提取标签
            labels = {
                'play_click': torch.tensor(batch['play_click'], dtype=torch.float),
                'play_1m_label': torch.tensor(batch['play_1m_label'], dtype=torch.float),
                'play_5m_label': torch.tensor(batch['play_5m_label'], dtype=torch.float),
                'play_9m_label': torch.tensor(batch['play_9m_label'], dtype=torch.float),
            }
            
            # 转换为torch tensors
            # discrete_tensor = torch.tensor(np.stack(discrete_features, axis=1), dtype=torch.long)
            # seq1_tensor = torch.tensor(np.stack(seq_features1, axis=1), dtype=torch.long)
            # seq2_tensor = torch.tensor(np.stack(seq_features2, axis=1), dtype=torch.long)
            # 连续特征可以根据实际情况添加
            # cont_tensor = torch.zeros(discrete_tensor.size(0), 0)  # 空的连续特征张量
            # labels_tensor = torch.tensor(np.stack(list(labels.values()), axis=1), dtype=torch.float)

            yield discrete_features, seq_features, labels

if __name__ == "__main__":
    files = get_file_list()
    print(f"Total files: {len(files)}") 
    cnt = 1
    print(f"{get_timestamp()} Start reading data...")
    dataset = StreamingTFRecordDataset(files, 0, 1, batch_size=16)
    for discrete_features, seq_features, labels in dataset:
        print(f"{get_timestamp()} Reading data: {cnt}")
        
        print("===========================discrete_features=================================")
        print(discrete_features)
        print("===========================seq_features1=================================")
        print(seq_features)
        
        print("===========================labels=================================")
        print(labels)
        cnt += 1
        if cnt > 3:
            break
