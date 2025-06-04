from setuptools import setup, find_packages

setup(
    name="tasq_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "gym",
        "pybullet",
        "torch",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "train-tasq = tasq_rl.scripts.train:main",
            "test-tasq = tasq_rl.scripts.test:main"
        ]
    },
    author="Storm de Kam",
    description="RL and control code for a soft quadruped robot",
)

