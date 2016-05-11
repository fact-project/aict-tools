from setuptools import setup

setup(
    name='klaas',
    version='0.0.4',
    description='KLAssification And regression Scripts.  yay',
    url='https://github.com/fact-project/klaas',
    author='Kai Brügge',
    author_email='kai.bruegge@tu-dortmund.de',
    license='BEER',
    packages=[
        'klaas',
        'klaas.scripts',
    ],

    install_requires=[
        'pandas',           # in anaconda
        'numpy',            # in anaconda
        'matplotlib>=1.4',  # in anaconda
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'pyyaml',             # in anaconda
        'tables',           # needs to be installed by pip for some reason
        # 'hdf5',
        'click',
        'numexpr',
        'pytest', # also in  conda
        'sklearn-pandas',
        'joblib',
        'tqdm',
    ],
   zip_safe=False,
   entry_points={
    'console_scripts': [
        'train_separation_model = klaas.scripts.train_separation_model:main',
        'train_energy_regressor = klaas.scripts.train_energy_regressor:main',
        'apply_separation_model = klaas.scripts.apply_separation_model:main',
        'apply_regression_model = klaas.scripts.apply_regression_model:main',
    ],
  }
)
