from setuptools import setup, find_packages

setup(
    name="agentic_workflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx",
        "ollama",
        "rich",
        "pydantic",
        # Add other required libraries
    ],
    entry_points={
        'console_scripts': [
            'math_workflow=agentic_workflow.main:main',
        ],
    },
)
