from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Deep_learning_projects",
    version="0.1.0",
    author="Farhan Fayaz",
    author_email="farhanfiaz79@gmail.com",
    description="A comprehensive template for deep learning projects with MLflow integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/farhan32742/deep_learning_project_template",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy<2.0.0",
        "pandas>=1.3.0",
        "tensorflow>=2.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "mlflow>=1.28.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "ultralytics"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "isort>=5.9.3",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-learning-train=Deep_learning_projects.pipeline.training_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
