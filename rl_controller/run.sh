counter=1
while [ $counter -le 100 ]
do
    roslaunch rl_controller start_rl_controller_st_line.launch
    killall -9 gzserver
    ((counter++))
done