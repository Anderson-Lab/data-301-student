#pip install wheel
#pip install jupytext --upgrade

git config --global core.editor "nano"

jupyter notebook --generate-config -y
echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py

# doc on notebook versioning
# https://nextjournal.com/schmudde/how-to-version-control-jupyter#3.3.-jupytext
