"""
setup.py — XAI-SplitShield installable package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="xai_splitshield",
    version="1.0.0",
    author="Anonymous Authors",
    description="XAI-SplitShield: Explainability-Driven Mitigation of Backdoor Attacks in Split Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/xai-splitshield",
    packages=find_packages(exclude=["tests*", "notebooks*", "results*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "xai-splitshield=experiments.run_experiment:main",
        ]
    },
)
