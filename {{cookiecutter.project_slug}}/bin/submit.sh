#!/bin/bash

EXP=$1
MESSAGE=$2

CV_SCORE=$(cat data/model/$EXP/cv_score.txt)


kaggle competitions submit \
    -c {{cookiecutter.competition_url_name}} \
    -f data/submit/$EXP/submission.csv \
    -m "$EXP(CV$CV_SCORE): $MESSAGE"
