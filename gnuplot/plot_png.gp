set terminal png large transparent
# perf1: compare performance for 3D/2D, non-OpenMP vs OpenMP and different CPUs
set output "../images/perf_flips.png"
set style fill solid
set style data boxes
set boxwidth 0.04
set xlabel "Number of spins"
set x2label "System size L (3D)"
set xrange [10:1e9]
set x2range [2.154:1e3]
set logscale xx2
set xtics 100,10,1e8
set x2tics 4,2,512
set ylabel "Spin updates per ns per GPU"
set grid y
set key above box
plot \
"haswell3d1GPUnoOMP.dat" u (($1)**3*0.75):2 t "Core i7-4770K + GeForce GTX TITAN (3D)", \
"haswell3d1GPU.dat" u (($1)**3*0.85):2 lt 9 t "Core i7-4770K + GeForce GTX TITAN (3D, OpenMP with 1 GPU)", \
"haswell2d1GPUnoOMP.dat" u (($1)**2):2 t "Core i7-4770K + GeForce GTX TITAN (2D)", \
"tcbs233d1GPUnoOMP.dat" u (($1)**3*0.95):2 t "Opteron 6376 + GeForce GTX TITAN (3D)", \
"tcbs233d2GPU.dat" u (($1)**3)*1.05:($2/2.0) t "Opteron 6376 + GeForce GTX TITAN (3D, OpenMP with 2 GPUs)",\
"grizzly3d1GPU.dat" u (($1)**3*1.15):2 lt 2 t "Xeon X5650 + Tesla C2075 (3D)", \
"grizzly3d4GPU.dat" u (($1)**3*1.25):($2/4.0) t "Xeon X5650 + Tesla C2075 (3D, OpenMP with 4 GPUs)"
reset

# perf2: runtime in hours for 10^6 spin sets and one coupling set
set output "../images/perf_time3d.png"
set style fill solid
set style data boxes
set boxwidth 0.04
set xlabel "System size L"
set logscale xy
set xtics 4,2,512
set xrange [2:1024]
set ylabel "runtime per set of couplings/ hours"
set grid y
set key above box
plot \
"grizzly3d4GPU.dat" u ($1*0.92):(($1)**3*$3/$2/1e3/3600) t "Xeon X5650 + Tesla C2075 (3D, N=6, OpenMP with 4 GPUs)", \
"tcbs233d2GPU.dat" u ($1*1.0):(($1)**3*$3/$2/1e3/3600) lt 9 t "Opteron 6376 + GF GTX TITAN (3D, N=6, OpenMP with 2 GPUs)", \
"haswell3d1GPUnoOMP.dat" u ($1*1.08):(($1)**3*$3/$2/1e3/3600) t "Core i7-4770K + GF GTX TITAN (3D, N=6, 1 GPU)"
reset

set output "../images/perf_time2d.png"
set style fill solid
set style data boxes
set boxwidth 0.04
set xlabel "System size L"
set logscale xy
set xtics 4
set xrange [2:32768]
set ylabel "runtime per set of couplings/ hours"
set grid y
set key above box
plot \
"grizzly2d4GPU.dat" u ($1*0.92):(($1)**2*$3/$2/1e3/3600) t "Xeon X5650 + Tesla C2075 (2D, N=6, OpenMP with 4 GPUs)", \
"tcbs232d2GPU.dat" u ($1*1.0):(($1)**2*$3/$2/1e3/3600) lt 9 t "Opteron 6376 + GF GTX TITAN (2D, N=6, OpenMP with 2 GPUs)", \
"haswell2d1GPUnoOMP.dat" u ($1*1.08):(($1)**2*$3/$2/1e3/3600) t "Core i7-4770K + GF GTX TITAN (2D, N=6, 1 GPU)"

reset
set output

