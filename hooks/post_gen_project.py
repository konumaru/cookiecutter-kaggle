import os
import shutil

project_dir = os.path.realpath(os.path.curdir)

old_path = os.path.join(project_dir, "_devcontainer")
new_path = os.path.join(project_dir, ".devcontainer")

if os.path.exists(old_path):
    shutil.move(old_path, new_path)
