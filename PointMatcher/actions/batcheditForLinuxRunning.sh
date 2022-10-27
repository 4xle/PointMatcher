# the modules are all referenced for building the Windows binary, probably?
# or maybe newer python (3.9.2) doesn't like the referencing style with the parent path?
# either way, these commands make pyhon3 __main__.py actually work on Linux

# need to have csvkit installed: 

# get all the files that have the PointMatcher path in them
grep -r 'from PointMatcher.*'| csvcut -d":" | csvsql -H -I --query 'SELECT DISTINCT a from stdin;' | csvformat -K 1 > editthesefiles.csv
# replace it
xargs -a editthesefiles.csv -I @@ -P 1 bash -c "sed -i -e 's/from PointMatcher\.*/from /g' @@;"
# in cases where the module itself was referenced, it appears to be for the __appname__ property
# that has been moved to the initVars.py file b/c that actually works
xargs -a editthesefiles.csv -I @@ -P 1 bash -c "sed -i -e 's/from  import\.*/from initVars import /g' @@;"
