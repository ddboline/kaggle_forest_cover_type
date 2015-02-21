#!/bin/bash

sudo apt-get install -y ipython python-matplotlib python-sklearn python-pandas htop

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_forest_cover_type/*.csv .

./my_model.py
