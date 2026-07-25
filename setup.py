"""
Setup configuration for addis-housing-price-predictor package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="addis-housing-price-predictor",
    version="1.0.0",
    author="Samuel Kahsay",
    author_email="samkahsay.dev@gmail.com",
    description="Synthetic housing data generator for Addis Ababa, Ethiopia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ethioel/addis-housing-price-predictor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-housing-data=src.data_generator:main",
        ],
    },
)
