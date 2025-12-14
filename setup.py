from setuptools import setup, find_packages

setup(
    name="babappalign",
    version="1.0.0",
    description="Deep learning–based progressive multiple sequence alignment engine",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Krishnendu Sinha",
    url="https://github.com/sinhakrishnendu/BABAPPAlign",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "biopython",
        "tqdm",
        "torch",
        "fair-esm",
    ],
    entry_points={
        "console_scripts": [
            "babappalign=babappalign.cli:main",
            "babappascore=babappalign.cli:score",
        ]
    },
)
