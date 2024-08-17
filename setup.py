from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_req(file_path:str) -> List[str]:
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [r.replace("\n", " ") for r in req]
        if HYPHEN_E_DOT in req:
            req.remove(HYPHEN_E_DOT)
    return req
    
setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Ayaan',
    author_email = 'ayaanrzaveri@gmail.com',
    packages = find_packages(),
    install_requires = ['numpy','pandas','seaborn']
)