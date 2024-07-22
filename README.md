The two files in this program both compare run times of various methods for matrix multiplication. tiledmm.c compares traditional matrix multiplication with tiled matrix multiplication, 
which decreases the size of the matrix multiplication into computation that can fit in the cache memory. threadmm.c compares tiled matrix multiplication with threaded matrix multiplication, 
where multipled threads simulatenously carry out the computation of the resultant matrix. This, when tested in execution, is the most efficient method.
