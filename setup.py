from setuptools import setup, find_packages

setup(
    name='ai_snake',
    version='0.1.0',
    description='AI Snake Game',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],  # You can add requirements here or use requirements.txt
    python_requires='>=3.7',
) 