#!/bin/bash
while [ "true" ]
do
    cd /home2/dungnguyen/length-controllable-summarisation
    source venv/bin/activate 
    python service_summarisation.py
    sleep 5
done
