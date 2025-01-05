from setuptools import setup, find_packages

setup(
    name='multi-armed-bandit',
    version='0.1.0',
    description='Implementations of Multi-Armed Bandit algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kirill Kazakov',
    author_email='kazakovkirill.mail@gmail.com',
    url='https://github.com/KazakovKB/multi-armed-bandit',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',  # Требуемая версия Python
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        'ipywidgets',
        'ipython',
    ],
    extras_require={
        'notebooks': ['jupyter', 'ipython'],
    },
    include_package_data=True,
)