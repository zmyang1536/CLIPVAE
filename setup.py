from setuptools import setup, find_packages

setup(
    name="clipvae",
    version="0.1.0",
    description="CLIPVAE: Contrastive Learning-Informed Variational Autoencoder for Multi-Modal Spatial Transcriptomics Analysis",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/CLIPVAE",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scanpy>=1.8.0",
        "anndata>=0.8.0",
        "pyyaml>=6.0",
        "argparse>=1.4.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.3.0",
        "umap-learn>=0.5.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)