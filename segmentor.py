import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import tensorflow.keras.backend as K # type: ignore
from vit_keras import vit
import os
import glob
import json

# Make sure data_handler.py is in the same directory
from data_handler import data_generator

# --- 1. Configuration Constants ---
# Data parameters
JSON_DIR = "data/sa-1b-000001-jsons/"  
IMAGE_DIR = "data/sa-1b-000001-images/"   

# Model & Training parameters
IMAGE_SIZE = (512, 512)
MASK_SIZE = (256, 256)
BATCH_SIZE = 4  # Adjust based on your GPU memory
NUM_POINTS_PROMPT = 10 # 1 positive, 9 negative
EPOCHS = 5

# Architecture parameters
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_DECODER_BLOCKS = 4


# --- 2. Model Definition ---

def create_image_encoder(image_size=(512, 512), trainable=False):
    """
    Creates the ViT-based image encoder.
    """
    base_model = vit.vit_b16(
        image_size=image_size,
        include_top=False,
        pretrained_top=False,
        weights="imagenet21k+imagenet2012",
        pretrained=True,
    )
    base_model.trainable = trainable

    inputs = base_model.input
    # The output of the ViT's encoder block is a sequence of embeddings
    feature_map = base_model.get_layer('Transformer_encoder_norm').output
    # We need to reshape it to have spatial dimensions for the decoder
    # The output shape is (batch_size, 1025, 768), we reshape the 1025 tokens.
    # For a 512x512 image, ViT-B/16 creates 32x32=1024 patches + 1 class token.
    # We will use the patch tokens.
    # Note: The output of ViT is different from CNNs. It's a sequence of patch embeddings.
    # For simplicity in the decoder, we will just use this sequence directly.
    encoder = tf.keras.Model(inputs, feature_map, name="image_encoder")
    return encoder

class PromptEncoder(layers.Layer):
    def __init__(self, embedding_dim=EMBEDDING_DIM, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.label_embedding = layers.Embedding(
            input_dim=3, 
            output_dim=embedding_dim, 
            name="label_embedding"
        )
        self.point_projection = layers.Dense(embedding_dim, name="point_projection")

    def call(self, points, labels):
        # Add 1 to labels to handle the -1 padding case (pad=0, bg=1, fg=2)
        labels_for_embedding = labels + 1
        point_embeddings = self.point_projection(tf.cast(points, tf.float16))
        label_embeddings = self.label_embedding(labels_for_embedding)
        prompt_embeddings = point_embeddings + label_embeddings
        return prompt_embeddings

class MaskDecoder(layers.Layer):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_blocks=NUM_DECODER_BLOCKS, image_embedding_patches=1024, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.decoder_blocks = [self.build_decoder_block() for _ in range(num_blocks)]
        
        num_patches_side = int(image_embedding_patches**0.5)
        self.upscale_projection = layers.Dense(num_patches_side * num_patches_side, name="upscale_projection")
        self.upscale_reshape = layers.Reshape((num_patches_side, num_patches_side, 1), name="upscale_reshape")
        
        self.upsampling_layers = keras.Sequential([
            layers.Conv2DTranspose(embedding_dim // 2, kernel_size=2, strides=2),
            layers.LayerNormalization(),
            layers.Activation("gelu"),
            layers.Conv2DTranspose(embedding_dim // 4, kernel_size=2, strides=2),
            layers.Activation("gelu"),
            layers.Conv2DTranspose(1, kernel_size=1)
        ], name="mask_upsampler")

    def build_decoder_block(self):
        return {
            "self_attention": layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim),
            "cross_attention": layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim),
            "ffn": keras.Sequential([layers.Dense(self.embedding_dim * 4, activation="gelu"), layers.Dense(self.embedding_dim)]),
            "norm1": layers.LayerNormalization(),
            "norm2": layers.LayerNormalization(),
            "norm3": layers.LayerNormalization(),
        }

    def call(self, image_embeddings, prompt_embeddings):
        x = prompt_embeddings
        for block in self.decoder_blocks:
            x = block["norm1"](x + block["self_attention"](query=x, value=x, key=x))
            attn_output = block["cross_attention"](query=x, value=image_embeddings, key=image_embeddings)
            x = block["norm2"](x + attn_output)
            x = block["norm3"](x + block["ffn"](x))

        mask_token_embedding = x[:, 0, :]
        
        upscaled_embedding = self.upscale_projection(mask_token_embedding)
        reshaped_embedding = self.upscale_reshape(upscaled_embedding)
        
        predicted_mask = self.upsampling_layers(reshaped_embedding)
        return tf.image.resize(predicted_mask, MASK_SIZE)

class Losses:
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coef

    @staticmethod
    def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        bce = K.binary_crossentropy(y_true_f, y_pred_f)
        p_t = (y_true_f * y_pred_f) + ((1 - y_true_f) * (1 - y_pred_f))
        alpha_factor = y_true_f * alpha + (1 - y_true_f) * (1 - alpha)
        modulating_factor = K.pow((1.0 - p_t), gamma)
        return K.mean(alpha_factor * modulating_factor * bce)

    @staticmethod
    def combined_loss(y_true, y_pred):
        f_loss = Losses.focal_loss(y_true, y_pred)
        d_loss = Losses.dice_loss(y_true, y_pred)
        return f_loss + d_loss
    
def build_segmentor_model(image_size=IMAGE_SIZE, num_points=NUM_POINTS_PROMPT):
    image_inputs = layers.Input(shape=(*image_size, 3), name="image_input")
    points_inputs = layers.Input(shape=(num_points, 2), dtype=tf.float16, name="points_input")
    labels_inputs = layers.Input(shape=(num_points,), dtype=tf.int32, name="labels_input")

    image_encoder = create_image_encoder(image_size=image_size, trainable=False)
    prompt_encoder = PromptEncoder()
    num_patches = (image_size[0] // 16) ** 2
    mask_decoder = MaskDecoder(image_embedding_patches=num_patches)

    image_embeddings_raw = image_encoder(image_inputs)
    image_embeddings = image_embeddings_raw[:, 1:, :]
    
    prompt_embeddings = prompt_encoder(points_inputs, labels_inputs)
    predicted_mask = mask_decoder(image_embeddings, prompt_embeddings)

    model = keras.Model(inputs=[image_inputs, points_inputs, labels_inputs], outputs=predicted_mask, name="segmentor_model")
    return model

if __name__ == "__main__":
    # Find all JSON files
    json_files = glob.glob(os.path.join(JSON_DIR, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in the specified directory: {JSON_DIR}")
    
    print(f"Found {len(json_files)} JSON files.")

    # Calculate total number of samples for steps_per_epoch
    total_annotations = 0
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            total_annotations += len(data['annotations'])

    print(f"Total annotations (samples) found: {total_annotations}")
    steps_per_epoch = total_annotations // BATCH_SIZE

    # Build and compile the model
    segmentor = build_segmentor_model()
    segmentor.summary()
    segmentor.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=Losses.combined_loss,
        metrics=['accuracy']
    )

    # Create the data generator
    train_generator = data_generator(
        json_file_list=json_files,
        image_dir=IMAGE_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        mask_size=MASK_SIZE,
        num_points=NUM_POINTS_PROMPT
    )

    # Start training
    print("\n--- Starting Training ---")
    segmentor.fit(
        train_generator,
        epochs=EPOCHS,
        # batch_size=BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    print("\n--- Training Finished ---")
    segmentor.save("segmentor_model.keras")