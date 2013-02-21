[ -z $3 ] && echo "USAGE: path_to/matlab file.hdf5 /ID" && exit
$1 -nodesktop -nosplash -r "h5read('$2','$3'), exit, %see also: h5disp('$2')"
