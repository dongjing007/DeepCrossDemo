import tensorflow as tf
from numpy import array

# 修改为自身的测试目录
training_samples_file_path = tf.keras.utils.get_file("test_data.csv","file:///Users/dj/Desktop/test/test_data.csv")

# load sample as tf dataset
# 1.batch_size 一次抓取样本数量 https://zhuanlan.zhihu.com/p/133864576
# 2.num_epochs 针对所有样本的一次迭代 https://www.cnblogs.com/bonelee/p/8383746.html
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12, 
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset

# 获取训练数据
train_dataset = get_dataset(training_samples_file_path)

# 所有分类特征
categorical_columns = []
# 物件id的emb特征 num_buckets 物件数
post_col = tf.feature_column.categorical_column_with_identity(
    key='post_id', num_buckets=165)
post_emb_col = tf.feature_column.embedding_column(post_col, 10)
categorical_columns.append(post_emb_col)

# 用户id的emb特征 num_buckets 用户数 categorical_column_with_hash_bucket 非连续使用此函数
user_col = tf.feature_column.categorical_column_with_identity(
    key='user_id', num_buckets=400)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
categorical_columns.append(user_emb_col)

# 所有数字特征
numerical_columns = [
    tf.feature_column.numeric_column('user_score_count', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('user_score_avg', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32), 
    tf.feature_column.numeric_column('user_score_stddev', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_score_count', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_score_avg', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_score_stddev', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_style', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_pattern', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_kind', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_size', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_budget_interval', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_size_interval', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_views', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32),
    tf.feature_column.numeric_column('post_weekly_views', shape=(1,),
                                     default_value=None,
                                     dtype=tf.dtypes.float32)
]

# embedding + MLP model 构建
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 編譯模型、設置損失函數、優化器和評估指標
# 1.binary_crossentropy 二元交叉熵
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# 训练模型
model.fit(train_dataset, epochs=5)

# 保存模型
tf.keras.models.save_model(
    model,
    "file:///Users/dj/Desktop/test/my_new_model",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)