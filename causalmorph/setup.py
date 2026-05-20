"""
Setup script for CausalMorph package
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="causalmorph",
    version="0.1.0",
    author="Mario De Los Santos-Hernández",
    author_email="madlsh3517@gmail.com",
    description="Preconditioning Data for Linear Non-Gaussian Acyclic Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarSH-Up/CausalMorph",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
