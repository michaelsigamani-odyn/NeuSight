from setuptools import setup, find_packages

NAME = "neusight"
PACKAGES = find_packages()
PYTHON_REQUIRES = ">=3.8"
INSTALL_REQUIRES = [
    "torch",
    "pandas",
    "tensorboard",
    "transformers==4.38.1",
    "pandas",
    "numpy",
    "tqdm",
    "torchvision",
    "scipy",
]

if __name__ == "__main__":
    setup(
        name=NAME,
        python_requires=PYTHON_REQUIRES,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
    )
