import json

import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessorFast,
    CLIPTextModel,
    CLIPTokenizerFast,
    CLIPVisionModel,
)

model_name = "openai/clip-vit-base-patch32"
processor = CLIPImageProcessorFast.from_pretrained(model_name)
image_model = CLIPVisionModel.from_pretrained(model_name)
image_model.to("cuda")

tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
text_model = CLIPTextModel.from_pretrained(model_name)
text_model.to("cuda")

print("Models and processors loaded successfully.")


def encode_image(image_paths: list[str] | str):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    imgs = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = image_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    latent_vectors = last_hidden_states[:, 0, :]
    return [latent_vectors[i] for i in range(latent_vectors.shape[0])]


def encode_text(texts: list[str] | str):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    latent_vectors = last_hidden_states[:, 0, :]
    return [latent_vectors[i] for i in range(latent_vectors.shape[0])]


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)

        self.combine_heads = tf.keras.layers.Dense(embed_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.1)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        output = self.combine_heads(concat_attention)
        output = self.dropout(output, training=training)

        # Residual connection + layer norm
        output = self.layer_norm(output + inputs)

        return output


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.1)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def call(self, query_input, key_value_input, training=False):
        batch_size = tf.shape(query_input)[0]

        query = self.query_dense(query_input)
        key = self.key_dense(key_value_input)
        value = self.value_dense(key_value_input)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        output = self.combine_heads(concat_attention)
        output = self.dropout(output, training=training)

        # Residual connection + layer norm
        output = self.layer_norm(output + query_input)

        return output, weights


class LocalizationLayer(tf.keras.layers.Layer):
    def __init__(self, image_dim=768, text_dim=512, target_dim=512, **kwargs):
        super(LocalizationLayer, self).__init__(**kwargs)

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.target_dim = target_dim

        # Projection layers to make dimensions consistent
        self.image_projection = tf.keras.layers.Dense(target_dim, name="image_proj")
        self.text_projection = tf.keras.layers.Dense(target_dim, name="text_proj")

        # Initialize all attention layers
        self.caption_self_attn = MultiHeadSelfAttention(
            embed_dim=target_dim, num_heads=8
        )
        self.image_caption_cross_attn = CrossAttention(
            embed_dim=target_dim, num_heads=8
        )
        self.image_prompt_cross_attn = CrossAttention(embed_dim=target_dim, num_heads=8)
        self.prompt_caption_cross_attn = CrossAttention(
            embed_dim=target_dim, num_heads=8
        )
        self.final_self_attn = MultiHeadSelfAttention(embed_dim=target_dim, num_heads=8)

        # Deep MLP layers for better "thinking" capacity
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dense3 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.dense4 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout4 = tf.keras.layers.Dropout(0.2)
        self.dense5 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout5 = tf.keras.layers.Dropout(0.2)
        self.dense6 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout6 = tf.keras.layers.Dropout(0.1)

        # Spatial reasoning layers - gradually expand to spatial dimensions
        self.spatial1 = tf.keras.layers.Dense(1024, activation="relu")
        self.spatial_dropout1 = tf.keras.layers.Dropout(0.2)
        self.spatial2 = tf.keras.layers.Dense(2048, activation="relu")
        self.spatial_dropout2 = tf.keras.layers.Dropout(0.2)
        self.spatial3 = tf.keras.layers.Dense(4096, activation="relu")
        self.spatial_dropout3 = tf.keras.layers.Dropout(0.1)

        # Final output layer
        self.output_dense = tf.keras.layers.Dense(224 * 224, activation="sigmoid")
        self.reshape_layer = tf.keras.layers.Reshape((224, 224, 1))

    def call(self, inputs, training=False):
        # Separate different modalities
        image_features_raw = tf.expand_dims(
            inputs[:, 0, : self.image_dim], 1
        )  # (batch, 1, 768)
        caption_features_raw = inputs[:, 1:6, : self.text_dim]  # (batch, 5, 512)
        prompt_features_raw = tf.expand_dims(
            inputs[:, 6, : self.text_dim], 1
        )  # (batch, 1, 512)

        # Project to consistent dimensions
        image_features = self.image_projection(image_features_raw)  # (batch, 1, 512)
        caption_features = self.text_projection(caption_features_raw)  # (batch, 5, 512)
        prompt_features = self.text_projection(prompt_features_raw)  # (batch, 1, 512)

        # Self-attention within each modality group
        attended_captions = self.caption_self_attn(caption_features, training=training)

        # Cross-attention: Image attending to captions
        image_attended, _ = self.image_caption_cross_attn(
            image_features, attended_captions, training=training
        )

        # Cross-attention: Image attending to prompt
        image_prompt_attended, _ = self.image_prompt_cross_attn(
            image_attended, prompt_features, training=training
        )

        # Cross-attention: Prompt attending to captions
        prompt_attended, _ = self.prompt_caption_cross_attn(
            prompt_features, attended_captions, training=training
        )

        # Combine all attended features
        combined_features = tf.concat(
            [
                image_prompt_attended,  # (batch, 1, 512)
                prompt_attended,  # (batch, 1, 512)
                attended_captions,  # (batch, 5, 512)
            ],
            axis=1,
        )  # Shape: (batch, 7, 512)

        # Final self-attention over all modalities
        final_features = self.final_self_attn(combined_features, training=training)

        # Global average pooling to get single vector
        pooled_features = tf.reduce_mean(final_features, axis=1)  # Shape: (batch, 512)

        # Deep MLP for feature processing and "thinking"
        x = self.dense1(pooled_features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.dense4(x)
        x = self.dropout4(x, training=training)
        x = self.dense5(x)
        x = self.dropout5(x, training=training)
        x = self.dense6(x)
        x = self.dropout6(x, training=training)

        # Spatial reasoning layers - gradually expand towards spatial dimensions
        x = self.spatial1(x)
        x = self.spatial_dropout1(x, training=training)
        x = self.spatial2(x)
        x = self.spatial_dropout2(x, training=training)
        x = self.spatial3(x)
        x = self.spatial_dropout3(x, training=training)

        # Final output layer - reshape to spatial dimensions
        outputs = self.output_dense(x)
        outputs = self.reshape_layer(outputs)

        return outputs


def create_localization_model():
    # Input: Mixed dimensions - image(768) + 5*captions(512) + prompt(512)
    # We'll use the maximum dimension (768) and pad/slice as needed
    inputs = tf.keras.Input(
        shape=(7, 768), name="input_features"
    )  # 1 image + 5 captions + 1 prompt, max 768 dims

    # Apply the localization layer with dimension specifications
    outputs = LocalizationLayer(image_dim=768, text_dim=512, target_dim=512)(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="localization_model")
    return model


def prepare_model_input(image_path, captions, prompt):
    """Prepare input tensor for the TensorFlow model from PyTorch CLIP encodings"""
    # Encode image (1 vector - 768 dims)
    image_vector = encode_image(image_path)[0]  # Shape: (768,)

    # Encode captions (5 vectors - 512 dims each)
    caption_vectors = encode_text(captions)  # List of 5 tensors, each (512,)

    # Encode prompt (1 vector - 512 dims)
    prompt_vector = encode_text(prompt)[0]  # Shape: (512,)

    # Convert PyTorch tensors to numpy
    image_np = image_vector.cpu().numpy()  # Shape: (768,)
    caption_np = torch.stack(caption_vectors).cpu().numpy()  # Shape: (5, 512)
    prompt_np = prompt_vector.cpu().numpy()  # Shape: (512,)

    # Pad text features to match image dimension (768) for consistent input shape
    max_dim = 768
    caption_padded = np.pad(
        caption_np, ((0, 0), (0, max_dim - 512)), "constant"
    )  # (5, 768)
    prompt_padded = np.pad(prompt_np, (0, max_dim - 512), "constant")  # (768,)

    # Combine into single input tensor
    combined_input = np.concatenate(
        [
            image_np[np.newaxis, :],  # Shape: (1, 768)
            caption_padded,  # Shape: (5, 768)
            prompt_padded[np.newaxis, :],  # Shape: (1, 768)
        ],
        axis=0,
    )  # Final shape: (7, 768)

    # Add batch dimension and convert to TensorFlow tensor
    model_input = tf.convert_to_tensor(
        combined_input[np.newaxis, :, :]
    )  # Shape: (1, 7, 768)

    return model_input


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for segmentation"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return 1 - dice


def combined_loss(y_true, y_pred):
    """Combined BCE + Dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def load_mask(mask_path):
    """Load and preprocess ground truth mask"""
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = mask.resize((224, 224))  # Resize to match model output
    mask_array = np.array(mask) / 255.0  # Normalize to [0, 1]
    # Convert to binary (0 or 1)
    mask_array = (mask_array > 0.5).astype(np.float32)
    return mask_array


def prediction_to_binary_mask(prediction, threshold=0.5, target_size=None):
    """Convert model prediction to binary mask (0 or 1)"""
    # prediction shape: (batch, 224, 224, 1)
    # Remove channel dimension and apply threshold
    binary_mask = (prediction.numpy().squeeze(-1) > threshold).astype(np.float32)

    # Resize to target size if specified
    if target_size is not None:
        resized_masks = []
        for i in range(binary_mask.shape[0]):  # For each batch item
            mask_image = Image.fromarray(
                (binary_mask[i] * 255).astype(np.uint8), mode="L"
            )
            resized_mask = mask_image.resize(target_size, Image.LANCZOS)
            # Convert back to binary after resizing
            resized_array = (np.array(resized_mask) / 255.0 > 0.5).astype(np.float32)
            resized_masks.append(resized_array)
        binary_mask = np.stack(resized_masks)

    return binary_mask


def get_image_size(image_path):
    """Get the size (width, height) of an image"""
    with Image.open(image_path) as img:
        return img.size


def resize_mask_to_original(mask, original_image_path):
    """Resize a mask to match the original image size using Lanczos"""
    target_size = get_image_size(original_image_path)

    # Convert mask to PIL Image
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    # Resize using Lanczos
    resized_mask = mask_image.resize(target_size, Image.LANCZOS)

    # Convert back to binary array
    resized_array = (np.array(resized_mask) / 255.0 > 0.5).astype(np.float32)

    print(
        f"Mask resized from {mask.shape} to {resized_array.shape} to match original image"
    )
    return resized_array


def evaluate_binary_metrics(y_true, y_pred, threshold=0.5):
    """Calculate IoU, Dice, and accuracy for binary segmentation"""
    # Convert predictions to binary
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > 0.5).astype(np.float32)

    # Flatten for calculations
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # Calculate metrics
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection

    # IoU (Intersection over Union)
    iou = intersection / (union + 1e-7)

    # Dice coefficient
    dice = (2 * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-7)

    # Pixel accuracy
    accuracy = np.mean(y_true_flat == y_pred_flat)

    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": accuracy,
        "Intersection": intersection,
        "Union": union,
    }


def save_prediction_mask(
    prediction, save_path, threshold=0.5, original_image_path=None
):
    """Save prediction as black and white image, optionally resized to original image size"""
    target_size = None

    # Get original image size if path provided
    if original_image_path:
        with Image.open(original_image_path) as img:
            target_size = img.size  # (width, height)
            print(f"Original image size: {target_size}")

    # Convert to binary mask with optional resizing
    binary_mask = prediction_to_binary_mask(prediction, threshold, target_size)

    # Convert to 0-255 range for image saving
    mask_image = (binary_mask[0] * 255).astype(np.uint8)  # Take first batch item

    # Save as image
    Image.fromarray(mask_image, mode="L").save(save_path)

    if target_size:
        print(f"Binary mask resized to {target_size} and saved to: {save_path}")
    else:
        print(f"Binary mask (224x224) saved to: {save_path}")

    return binary_mask


def create_tf_dataset(dataset, batch_size=8, shuffle=True, repeat=True):
    """Create a proper tf.data.Dataset for training"""

    def data_generator():
        while True:  # Infinite loop for repeating
            indices = list(range(len(dataset)))
            if shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_inputs = []
                batch_masks = []

                # Handle the last batch which might be smaller
                batch_end = min(i + batch_size, len(indices))
                current_batch_size = batch_end - i

                for j in range(current_batch_size):
                    idx = indices[i + j]
                    sample = dataset[idx]

                    try:
                        # Prepare model input
                        model_input = prepare_model_input(
                            sample["original_image"],
                            sample["captions"],
                            sample["prompt"],
                        )
                        batch_inputs.append(model_input[0])  # Remove batch dimension

                        # Load ground truth mask
                        mask_path = sample["mask"]
                        mask = load_mask(mask_path)
                        batch_masks.append(mask)

                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        # Skip this sample and continue
                        continue

                if len(batch_inputs) > 0:  # Only yield if we have valid data
                    # Pad batch if needed to maintain consistent batch size
                    while len(batch_inputs) < batch_size:
                        batch_inputs.append(batch_inputs[-1])  # Duplicate last sample
                        batch_masks.append(batch_masks[-1])

                    yield (
                        tf.stack(batch_inputs[:batch_size]),
                        tf.expand_dims(tf.stack(batch_masks[:batch_size]), -1),
                    )

            if not repeat:
                break

    # Create tf.data.Dataset from generator
    dataset_tf = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 7, 768), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 224, 224, 1), dtype=tf.float32),
        ),
    )

    return dataset_tf


def train_model(model, train_dataset, val_dataset, epochs=20, batch_size=8):
    """Training function"""
    print(f"\nStarting training with {len(train_dataset)} training samples...")

    # Create tf.data.Dataset objects
    print("Creating training dataset...")
    train_ds = create_tf_dataset(train_dataset, batch_size, shuffle=True, repeat=True)

    print("Creating validation dataset...")
    val_ds = create_tf_dataset(val_dataset, batch_size, shuffle=False, repeat=True)

    # Calculate steps per epoch
    train_steps = max(1, len(train_dataset) // batch_size)
    val_steps = max(1, len(val_dataset) // batch_size)

    # Ensure we have at least some steps
    if train_steps == 0:
        train_steps = 1
    if val_steps == 0:
        val_steps = 1

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    print(f"Training batch size: {batch_size}")
    print(f"Total training batches per epoch: {train_steps}")
    print(f"Total validation batches per epoch: {val_steps}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_localization_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
        ),
    ]

    # Train the model
    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    return history


if __name__ == "__main__":
    import numpy as np

    print("Loading dataset...")

    # Load dataset
    with open("data_mappings.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total dataset size: {len(dataset)} samples")

    # Simple train/validation split (80/20)
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create and compile model
    print("\nCreating model...")
    model = create_localization_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=["accuracy"],
    )

    print("\nModel Summary:")
    model.summary()

    # Test with one sample first
    print("\nTesting with one sample...")
    sample = train_dataset[0]
    model_input = prepare_model_input(
        sample["original_image"], sample["captions"], sample["prompt"]
    )

    print(f"Input shape: {model_input.shape}")

    # Test forward pass
    prediction = model(model_input, training=False)
    print(f"Output shape: {prediction.shape}")
    print(
        f"Output range: [{prediction.numpy().min():.4f}, {prediction.numpy().max():.4f}]"
    )

    # Convert to binary mask and save
    print("\nConverting to binary mask...")
    binary_mask = prediction_to_binary_mask(prediction, threshold=0.5)
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Binary mask values: unique = {np.unique(binary_mask)}")
    print(
        f"Binary mask stats: {np.sum(binary_mask[0])} white pixels out of {224 * 224} total"
    )

    # Save the binary mask as an image (224x224)
    save_prediction_mask(prediction, "test_prediction_mask_224.png", threshold=0.5)

    # Save the binary mask resized to original image size
    save_prediction_mask(
        prediction,
        "test_prediction_mask_original_size.png",
        threshold=0.5,
        original_image_path=sample["original_image"],
    )

    # Compare with ground truth if available
    if "mask" in sample:
        print("\nEvaluating against ground truth...")
        gt_mask = load_mask(sample["mask"])

        # Evaluate at 224x224 resolution (model's native output)
        metrics_224 = evaluate_binary_metrics(
            gt_mask, prediction.numpy().squeeze(-1)[0]
        )
        print("\nMetrics at 224x224:")
        print(f"IoU Score: {metrics_224['IoU']:.4f}")
        print(f"Dice Score: {metrics_224['Dice']:.4f}")
        print(f"Pixel Accuracy: {metrics_224['Accuracy']:.4f}")

        # Also evaluate at original image size
        original_size = get_image_size(sample["original_image"])
        print(f"\n🔍 Original image size: {original_size}")

        # Resize prediction to original size for comparison
        binary_mask_original = prediction_to_binary_mask(
            prediction, threshold=0.5, target_size=original_size
        )

        # Load ground truth at original size
        gt_mask_original = Image.open(sample["mask"]).convert("L")
        gt_mask_original = (np.array(gt_mask_original) / 255.0 > 0.5).astype(np.float32)

        metrics_original = evaluate_binary_metrics(
            gt_mask_original, binary_mask_original[0]
        )
        print(f"\nMetrics at original size {original_size}:")
        print(f"IoU Score: {metrics_original['IoU']:.4f}")
        print(f"Dice Score: {metrics_original['Dice']:.4f}")
        print(f"Pixel Accuracy: {metrics_original['Accuracy']:.4f}")

        # Save ground truth for comparison (224x224)
        gt_image = (gt_mask * 255).astype(np.uint8)
        Image.fromarray(gt_image, mode="L").save("test_ground_truth_mask_224.png")
        print("\nGround truth mask (224x224) saved to: test_ground_truth_mask_224.png")

        # Save ground truth at original size
        gt_image_original = (gt_mask_original * 255).astype(np.uint8)
        Image.fromarray(gt_image_original, mode="L").save(
            "test_ground_truth_mask_original_size.png"
        )
        print(
            "Ground truth mask (original size) saved to: test_ground_truth_mask_original_size.png"
        )

    # Start training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    try:
        print("Starting training...")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=50,
            batch_size=4,
        )

        print("\nTraining completed successfully!")

        # Save final model
        model.save("final_localization_model.keras")
        print("Model saved as 'final_localization_model.keras'")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving current model state...")
        model.save("checkpoint_localization_model.keras")
        raise
