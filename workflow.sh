#!/bin/bash

cd /ws && source devel/setup.bash && roslaunch motion bringup.launch & sleep 15 \
&& cd /ws \
&& source devel/setup.bash \
&& rosrun motion ros.py \
&& rosrun motion library.py \
&& rosrun motion package.py \
&& rosrun motion sim.py
