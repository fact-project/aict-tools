from setuptools import setup, find_packages

setup(
    name='aict_tools',
    version='0.12.5',
    description='Artificial Intelligence for Imaging Atmospheric Cherenkov Telescopes',
    url='https://github.com/fact-project/aict-tools',
    author='Kai Brügge, Maximilian Nöthe, Jens Buss',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'astropy',
        'click',
        'h5py',
        'joblib',
        'matplotlib>=2.0',  # in anaconda
        'numexpr',
        'numpy',            # in anaconda
        'pandas',           # in anaconda
        'pyfact>=0.16.0',
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'pyyaml',             # in anaconda
        'scikit-learn==0.19.1',
        'sklearn2pmml',
        'tables',           # needs to be installed by pip for some reason
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
            'aict_plot_gh_performance = aict_tools.scripts.plot_gh_performance:main',
            'aict_plot_energy_performance = aict_tools.scripts.plot_energy_performance:main',
            'aict_plot_direction_performance = aict_tools.scripts.plot_direction_performance:main',
            'aict_apply_cuts = aict_tools.scripts.apply_cuts:main',
            'aict_convert_pandas2h5py = klaas.scripts.convert_pandas2h5py:main',
            'fact_to_dl3 = aict_tools.scripts.fact_to_dl3:main',
        ],
    }
)
