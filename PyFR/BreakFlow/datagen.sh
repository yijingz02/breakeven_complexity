#!/bin/bash

STAGING_PATH="."

# if [ -z $1 ] ; then
#   N=100
# else
#   N=$1
# fi

# ###########################  generates all meshes  ############################
# cd data
# python ../meshgen.py $N

START=0
END=100

if [ $# -ge 1 ]; then
  END="$1"        # bash datagen.sh 1000  -> 0 1000
fi

if [ $# -ge 2 ]; then
  START="$1"      # bash datagen.sh 1000 2000 -> 1000 2000
  END="$2"
fi

###########################  generates all meshes  ############################
cd data
python ../meshgen.py "$START" "$END"

# for SEED in `seq 0 $((N - 1))` ; do 
#   if [ ! -f $SEED/flow.h5 ] ; then

for SEED in $(seq "$START" $((END - 1))); do
  if [ ! -f "$SEED/flow_${SEED}.h5" ] ; then

##########################  converts to PyFR mesh  ############################
    cd $SEED
    cp ../../inc-flow.ini .
    pyfr import mesh.msh mesh.pyfrm

#########  simulation wrapped in loop that reduces dt if NaNs occur  ##########
    for T in 5 4 3 2 1 ; do
      rm -rf *.pyfrs
      sed -i 's/dt = 0.0.$/dt = 0.0'$T'/' inc-flow.ini
      sed -i 's/pseudo-dt = 0.00.$/pseudo-dt = 0.00'$T'/' inc-flow.ini
      # pyfr -p run -b cuda mesh.pyfrm inc-flow.ini
      pyfr -p run -b cuda mesh.pyfrm inc-flow.ini 1> /dev/null
      if [ ! $(tail -n 1 residual.csv | grep "nan") ] ; then
        break
      fi
    done

#####################  converts outputs to VTU format  ########################
    for PYFRS in *.pyfrs ; do
      pyfr export mesh.pyfrm $PYFRS ${PYFRS%.pyfrs}.vtu
    done

###################### interpolates onto regular grid  ########################
    python ../../flowgen.py 
    rm -rf *.vtu

    # Rename flow.h5 to include the seed number
    if [ -f flow.h5 ]; then
      mv -f flow.h5 "flow_${SEED}.h5"
    fi

    cp -r "flow_${SEED}.h5" "$STAGING_PATH"

    cd ..

  fi
done

cd ..
