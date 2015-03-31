#!/bin/bash

rm /home/ubuntu/.ssh/known_hosts
scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/backup_kaggle/kaggle_forest_cover_type/forest_cover_type.tar.gz .
tar zxvf forest_cover_type.tar.gz
rm forest_cover_type.tar.gz

./my_model.py $1

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk FOREST_COVER_DONE_$1"
