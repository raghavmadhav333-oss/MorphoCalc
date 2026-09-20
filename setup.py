from setuptools import setup, find_packages

setup(
    name="morpho_calculus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0"
    ],
    description="A Real-Time Neural Surrogate Engine for Chaotic Fluid Dynamics (PARC)",
    author="Raghav",
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'morpho-engine=morpho_calculus.dataset_engine:main',
            'morpho-decoder=morpho_calculus.symbolic_decoder:main',
        ],
    }
)
