rm log/params.pt
counter=1
while [ $counter -le 20000 ]
do
    source ../../../devel/setup.bash
    roslaunch rl_controller start_rl_controller_st_line.launch eps_no:=$counter
    killall -9 gzserver
    ((counter++))
done