#!/bin/bash
while [ "true" ]
do
    cd /home/kuuhaku/work/length-controllable-summarisation/en
    source /home/kuuhaku/work/length-controllable-summarisation/venv/bin/activate 
    python service.py
    sleep 5
done