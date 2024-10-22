#!/usr/bin/env python3

import os
import sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(file_path)

def copy_files(from_path , to_path , files = ['.python-version' , 'pyproject.toml' , 'requirements.lock' , 'requirements-dev.lock']):
    for fl in files:
        os.system(f"cp -r {file_path} {project_path}/{fl}")

# load environment variables from .envrc file if it exists , Load VENV_PATH from .envrc file
venv_path = None
if os.path.exists('.envrc'):
    with open('.envrc') as f:
        for line in f:
            if line.startswith('export VENV_PATH='):
                venv_path = line.split('=')[1].strip().replace('/.venv', '')[1:-1]
                
if venv_path is not None:
    print(f"Using VENV_PATH: {venv_path}\n-------------------------\n\n")
    # change the current working directory to the project directory
    # If the project directory does not exist, create it
    if not os.path.exists(venv_path):
        # Create project directory with parent directory
        print("VENV_PATH does not exist. Creating it...")
        os.makedirs(venv_path , exist_ok=True)
        
        
    os.chdir(venv_path)
    # print current working directory

# if sys.argv[-1] == '@force_replace':
#     argv = sys.argv[1:-1]
#     force_replace = True
#     print("Forcing replace...\n-------------------------\n\n")
# else:
#     force_replace = False
#     argv = sys.argv[1:]

force_replace = True
argv = sys.argv[1:]

input_argv = ' '.join(argv) 

os.system(f"rye {input_argv}")

file_ignore = ['.venv' , '.directory' , 'src' , 'README.md']
file_lists = os.listdir()
# remove .venv from file lists
file_lists = [f for f in file_lists if f not in file_ignore]
# print(file_lists)

# copy the file_lists into project directory if they do not exist
for fl in file_lists:
    file_path = os.path.join(str(venv_path), str(fl))
    if not os.path.exists(f'{project_path}/{fl}') or force_replace:
        if force_replace:
            os.system(f'rm -r {project_path}/{fl}')
        os.system(f"cp -r {file_path} {project_path}/{fl}")





print("\n# PUV DONE\n-------------------------")