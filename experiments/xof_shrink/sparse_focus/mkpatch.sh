#!/bin/bash
# copies the builder and tools into the repo copy and writes patch.diff (repo diff + new files)
A=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof2/av/sparse-focus
E=$A/repo/experiments/xof_shrink
cp $A/sp.py $E/sp.py
mkdir -p $E/sparse_focus
cp $A/layerstats.py $A/eager_check.py $A/collect.py $A/audit.sh $A/chi_search.py $A/par_search.py $A/mkpatch.sh $E/sparse_focus/
git -C $A/repo add -N experiments/xof_shrink/sp.py experiments/xof_shrink/sparse_focus
git -C $A/repo diff > $A/patch.diff
wc -l $A/patch.diff
# keep the working copy clean (the harness imports sp from this directory, not from the repo)
git -C $A/repo reset -q experiments/xof_shrink/sp.py experiments/xof_shrink/sparse_focus
rm -rf $E/sp.py $E/sparse_focus
