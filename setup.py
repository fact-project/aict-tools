from setuptools import setup, find_packages
from os import path

d = path.abspath(path.dirname(__file__))
with open(path.join(d, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aict_tools',
    version='0.16.0',
    description='Artificial Intelligence for Imaging Atmospheric Cherenkov Telescopes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fact-project/aict-tools',
    author='Kai Brügge, Maximilian Nöthe, Jens Buss',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'astropy',  # in anaconda
        'click',  # in anaconda
        'h5py',  # in anaconda
        'joblib',  # in anaconda
        'matplotlib>=2.0',  # in anaconda
        'numexpr',  # in anaconda
        'numpy',            # in anaconda
        'pandas',           # in anaconda
        'pyfact>=0.16.0',
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'pyyaml',             # in anaconda
        'scikit-learn~=0.20.0',  # See PEP 440, compatible releases
        'sklearn2pmml',
        'tqdm',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'aict_train_separation_model = aict_tools.scripts.train_separation_model:main',
            'aict_apply_separation_model = aict_tools.scripts.apply_separation_model:main',
            'aict_train_energy_regressor = aict_tools.scripts.train_energy_regressor:main',
            'aict_apply_energy_regressor = aict_tools.scripts.apply_energy_regressor:main',
            'aict_train_disp_regressor = aict_tools.scripts.train_disp_regressor:main',
            'aict_apply_disp_regressor = aict_tools.scripts.apply_disp_regressor:main',
            'aict_split_data = aict_tools.scripts.split_data:main',
            'aict_plot_separator_performance = aict_tools.scripts.plot_separator_performance:main',
            'aict_plot_regressor_performance = aict_tools.scripts.plot_regressor_performance:main',
            'aict_apply_cuts = aict_tools.scripts.apply_cuts:main',
            'aict_convert_pandas2h5py = aict_tools.scripts.convert_pandas2h5py:main',
            'fact_to_dl3 = aict_tools.scripts.fact_to_dl3:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
