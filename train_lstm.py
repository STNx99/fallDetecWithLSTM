import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Kiểm tra và cấu hình GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Đặt chế độ sử dụng bộ nhớ linh hoạt
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=10240)])  # Giới hạn bộ nhớ (nếu cần)
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs are available and configured.")
    except RuntimeError as e:
        print("Error configuring GPUs: ", e)
else:
    print("No GPUs found. Training will use CPU.")

# Đọc dữ liệu từ file
fall_df = pd.read_csv('falldetec.txt')
natural_df = pd.read_csv('natural.txt')

X = []
y = []
no_of_timesteps = 20

# Chuẩn bị dữ liệu cho training
datasets = natural_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

datasets = fall_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)

X, y = np.array(X), np.array(y)
print("Shape of X, y:", X.shape, y.shape)

# Chia tập dữ liệu thành training và testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training data shape:", X_train.shape, y_train.shape)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))

# Biên dịch mô hình
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Huấn luyện mô hình
with tf.device('/GPU:0'):  # Đảm bảo mô hình được huấn luyện trên GPU
    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))  # Giảm số epoch để thử nghiệm

# Lưu mô hình
model.save("model.bak.h5")
