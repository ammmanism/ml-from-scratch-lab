"""
Example script demonstrating K-Means for image compression.
"""

import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.metrics import mean_squared_error
from src.ml_from_scratch.unsupervised.kmeans import KMeans


def main():
    # Load a sample image (flower)
    try:
        image = load_sample_image("flower.jpg")
    except:
        # If flower.jpg is not available, create a simple test image
        image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    
    # Display original image info
    print(f"Original image shape: {image.shape}")
    print(f"Original image size (MB): {image.nbytes / 1024 / 1024:.2f}")
    
    # Reshape the image to be a list of pixels (each pixel is an RGB value)
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float64)
    
    # Apply K-Means clustering to compress the image
    n_colors = 16  # Number of colors to reduce to
    kmeans = KMeans(n_clusters=n_colors, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Replace each pixel with its corresponding cluster center
    compressed_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
    
    # Reshape back to image dimensions
    compressed_image = compressed_pixels.reshape(h, w, c)
    
    print(f"Compressed image shape: {compressed_image.shape}")
    print(f"Compressed image size (MB): {(compressed_image.nbytes + kmeans.cluster_centers_.nbytes) / 1024 / 1024:.2f}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print(f"Number of colors reduced from 256^3 to {n_colors}")
    
    # Calculate compression ratio
    original_size = image.nbytes
    compressed_size = compressed_image.nbytes + kmeans.cluster_centers_.nbytes  # Add cluster centers size
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}x")


if __name__ == "__main__":
    main()