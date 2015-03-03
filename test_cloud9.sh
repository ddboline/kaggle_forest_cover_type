#!/bin/bash

rm /home/ubuntu/.ssh/known_hosts
scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_forest_cover_type/*.csv .

./my_model.py

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE"
