from setuptools import setup, find_packages

setup(
    name="carla-test-platform",
    version="0.1.0",
    description="CARLA自动驾驶算法测试平台",
    author="Your Organization",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    extras_require={
        "dev": [
            line.strip()
            for line in open("requirements-dev.txt").readlines()
            if not line.startswith("#") and not line.startswith("-r")
        ]
    },
    entry_points={
        "console_scripts": [
            "carla-train=src.scripts.train:main",
            "carla-eval=src.scripts.evaluate:main",
            "carla-viz=src.scripts.visualize:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 