from setuptools import setup, find_packages

setup(
  name = 'nuwa-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.2.1',
  license='MIT',
  description = 'NÜWA - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/nuwa-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'einops>=0.3',
    'ftfy',
    'regex',
    'torch>=1.6',
    'torchvision',
    'unfoldNd',
    'vector-quantize-pytorch>=0.4.10'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
