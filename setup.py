from setuptools import setup, find_packages

setup(
    name="morphocalc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0"
    ],
    description="A Morpho-Computational Calculus framework to simplify and accelerate complex fluid dynamics.",
    author="Raghav",
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'morphocalc-engine=morphocalc.dataset_engine:main',
            'morphocalc-decoder=morphocalc.symbolic_decoder:main',
        ],
    }
)
