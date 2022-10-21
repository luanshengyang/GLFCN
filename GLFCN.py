from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import tensorflow as tf


def Attention(x, dim, num_heads, attn_drop=0.5, proj_drop=0.5):
    B, N, C = x.shape
    self_qkv = layers.Dense(dim * 3)(x)
    head_dim = dim // num_heads
    self_scale = head_dim ** -0.5
    qkv = K.permute_dimensions(K.reshape(self_qkv, [-1, N, 3, num_heads, C // num_heads]), (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ K.permute_dimensions(k, (0, 1, 3, 2))) * self_scale
    attn = tf.nn.softmax(attn, axis=-1)
    attn = layers.Dropout(attn_drop)(attn)
    x = K.reshape(K.permute_dimensions((attn @ v), (0, 2, 1, 3)), [-1, N, C])
    x = layers.Dense(dim)(x)
    x = layers.Dropout(proj_drop)(x)
    return x


# ---------------------------------------------------------------------------------------------------------------------
in_shp = []  # [2,128]
classes = mods  # 11

dr = 0.5  # dropout rate
# GLFCN ---------------------------------------------------------------------------------------------------------------
inputs = layers.Input(shape=in_shp)  # [N,2,128]
# stage1
out = layers.Reshape([1]+in_shp)(inputs)  # [N,1,2,128]
out = layers.ZeroPadding2D((0, 1), data_format="channels_first")(out)
out = layers.Conv2D(kernel_initializer="glorot_uniform", activation="relu", data_format="channels_first",
                    padding="valid", filters=256, kernel_size=(2, 3))(out)  # [N,256,2,128]
out = layers.Dropout(dr)(out)
# stage2
out = layers.Reshape([256, 256])(out)
out = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', data_format='channels_first',
                    kernel_initializer='glorot_uniform')(out)  # [N,256,128]
out = layers.Dropout(dr)(out)
# stage3
out = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', data_format='channels_first',
                    kernel_initializer='glorot_uniform')(out)  # [N,80,128]
out = layers.Dropout(dr)(out)

# stage4
out = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', data_format='channels_first',
                    kernel_initializer='glorot_uniform')(out)  # [N,80,128]
out = layers.Dropout(dr)(out)

# stage5
out_norm = layers.LayerNormalization(epsilon=1e-6)(inputs)  # [N,2,256]
attention_output = Attention(out_norm, dim=256, num_heads=8)  # [N,2,256]
out_trans = layers.Add()([out_norm, attention_output])  # [N,2,128]
out = layers.Concatenate(axis=1)([out, out_trans])  # [N,82,128]

out = layers.Flatten()(out)
out = layers.Reshape((1, 16896))(out)

# stage6
out = layers.Dense(256, activation='relu')(out)
out = layers.Dropout(dr)(out)
out = layers.Dense(len(classes))(out)
outputs = layers.Activation('softmax')(out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()



