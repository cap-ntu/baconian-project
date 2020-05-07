#!/usr/bin/env bash
# run the script locally to copy the log from remote server back
if (( $# != 3 )); then
    echo "Illegal number of parameters"
    exit 2
fi

docker=$1
env=$2
algo=$3

ssh 2080Ti << EOF
  rm -rf /home/dls/tmp && mkdir /home/dls/tmp && \
  mkdir /home/dls/tmp/$env/ && \
  mkdir /home/dls/tmp/$env/$algo && \
  docker cp $docker:/baconian-project/baconian/benchmark/benchmark_log/$env/$algo/. /home/dls/tmp/$env/$algo/ \
 && exit
EOF

mkdir -p /Users/lukeeeeee/Code/baconian-internal/baconian/benchmark/benchmark_log/$env && \
mkdir -p /Users/lukeeeeee/Code/baconian-internal/baconian/benchmark/benchmark_log/$env/$algo/ && \
scp -r 2080Ti:/home/dls/tmp/$env/$algo /Users/lukeeeeee/Code/baconian-internal/baconian/benchmark/benchmark_log/$env/