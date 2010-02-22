#! /bin/sh
python euler.py \
   'case=AirplaneProblem("../../meshpy/test/ka-6d.ply")' \
   quad_min_degree=0 \
   vis_interval_steps=1 \
   "$@"

