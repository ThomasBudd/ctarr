from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ctarr',
    url='https://github.com/ThomasBudd/ctarr',
    author='Thomas Buddenkotte',
    author_email='thomasbuddenkotte@googlemail.com',
    # Needed to actually package something
    packages=['ctarr'],
    # Needed for dependencies
    install_requires=[
            "torch>=1.7.0",
            "tqdm",
            "numpy",
            "nibabel",
      ],
    # *strongly* suggested for sharing
    version='1.0',
    description='toolbox for automated recognition of anatomical regions in CT images',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)