#!/bin/bash

function ProgressBar {
# Process data
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
# Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

# 1.2 Build progressbar strings and print the ProgressBar line
# 1.2.1 Output example:
# 1.2.1.1 Progress : [########################################] 100%
printf "\rRunning ${2} instances : [${_fill// /#}${_empty// /-}] ${_progress}%%"

}

let max=100
tmpfile=$(mktemp)

if [[ $# -ne 2 ]]
then
	echo -e "Usage : ${0} <algo_id> <size>"
	echo -e
	echo -e "Allowed algo_ids are :"
	echo -e "\t0 -> logger"
	echo -e "\t1 -> montecarlo"
	echo -e "\t2 -> genetic"
	echo -e "\t3 -> dumb"
	exit
fi

echo "propagator;branchers;runtime;solutions;propagations;nodes;failures;restarts;no-goods;peak depth" > ${tmpfile}

for (( i = 0; i <= $max; i++ )); do
    ProgressBar ${i} ${max}
    ../baker -algo ${1} -n ${2} 2> /dev/null | cut -d ':' -f 2 | grep -i "[0-9]" | sed -e 's/^ *\([0-9.]\+\).*/\1;/' | tr -d '\n' >> ${tmpfile} && echo "" >> ${tmpfile}
done
echo
./compute_results.py ${tmpfile} ${1} ${2}
mv ${tmpfile} results_${1}_${2}_gen.txt
echo -ne "\nDone !\n"
