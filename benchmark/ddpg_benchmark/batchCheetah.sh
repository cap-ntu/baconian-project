max=10
for i in `seq 1 $max`
do
    echo "$i"
    python3 halfCheetah.py "$i" &
done

