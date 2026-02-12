from setuptools import setup, find_packages

setup(
    name='ai_snake',
    version='0.1.0',
    description='AI Snake Game',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pygame>=2.6.1',
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'tqdm>=4.64.0',
        'PyYAML>=6.0',
    ],
    extras_require={
        'test': ['pytest>=7.0.0'],
    },
    entry_points={
        'console_scripts': [
            'snake=ai_snake.cli:main',
        ],
    },
    python_requires='>=3.8',
) 
