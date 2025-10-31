from PIL import Image
import hashlib
import io
import os

class DataLoader:
    def __init__(self, data_dir: str = "data/images"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _get_content_hash(self, image: Image.Image) -> str:
        """
        Calculates the SHA-256 hash of the image's raw pixel data.
        
        This hash is consistent for any image with the same content.
        """
        image_bytes = image.tobytes()
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        return hasher.hexdigest()

    def to_pil(self, image_path: str) -> Image.Image:
        """
        Load an image from the specified path and convert it to a PIL Image.

        Args:
            image_path (str): Path to the image file.
        Returns:
            Image.Image: The loaded PIL Image.
        """
        return Image.open(image_path).convert("RGB")
    
    def save_image(self, image: Image.Image):
        """
        Save the given PIL Image to the data directory.

        Args:
            image (Image.Image): The PIL Image to save.
        Returns:
            str: The path where the image was saved.
        """
        hash_name = self._get_content_hash(image) + ".png"
        save_path = os.path.join(self.data_dir, hash_name)
        if not os.path.exists(save_path):
            image.save(save_path)
        return save_path
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from the specified path.

        Args:
            image_path (str): Path to the image file.
        Returns:
            Image.Image: The loaded PIL Image.
        """
        pil_image = self.to_pil(image_path)
        hash_name = self._get_content_hash(pil_image) + ".png"
        if os.path.exists(os.path.join(self.data_dir, hash_name)):
            return Image.open(os.path.join(self.data_dir, hash_name)).convert("RGB")
        else:
            save_path = self.save_image(pil_image)
            return self.to_pil(save_path)
        
    def convert_to_blob(self, image: Image.Image, format: str = "PNG") -> bytes:
        """
        Convert a PIL Image to a binary blob.

        Args:
            image (Image.Image): The PIL Image to convert.
            format (str): The format to save the image in (default is "PNG").
        Returns:
            bytes: The binary blob of the image.
        """ 
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr