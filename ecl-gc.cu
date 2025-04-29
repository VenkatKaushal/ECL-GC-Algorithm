============================================
========== Header Inclusions and Constants ==========
============================================

#include <algorithm>  // For standard functions like max
#include <cuda.h>      // For CUDA runtime APIs
#include "ECLgraph.h"  // Graph input/output handling (provided separately)

// Device and Kernel configuration constants
static const int Device = 0;
static const int ThreadsPerBlock = 512; // CUDA thread block size
static const unsigned int Warp = 0xffffffff; // Full 32-bit warp mask
static const int WS = 32;  // Warp size (also bits in int)
static const int MSB = 1 << (WS - 1); // Most significant bit
static const int Mask = (1 << (WS / 2)) - 1; // 16-bit mask for half a word

// Device-wide global variable for worklist size
static __device__ int wlsize = 0;


============================================
========== Helper Functions ==========
============================================

// Fast hash function for integers (used to break symmetry between nodes)
static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


============================================
========== Initialization Kernel ==========
============================================
/*
__global__: This is a GPU kernel function that can be launched from CPU and executed on GPU.
static: Limits the scope of this function to the file (optional here).

Arguments:
nodes: number of vertices in the graph.
edges: number of edges.
nidx: adjacency list index array (nidx[v] points to start of neighbors for vertex v).
nlist: adjacency list containing neighbor vertices.
nlist2: workspace to store reordered neighbors based on priorities.
posscol: possible colors for each node (bitmap format).
posscol2: workspace array for possible colors.
color: stores assigned colors or metadata for each vertex.
wl: worklist (dynamic array to store vertices to be processed later).
__restrict__: Tells the compiler that pointers do not alias each other, allowing better optimization.

*/

// Kernel to initialize coloring metadata and worklists
static __global__
void init(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist,
          int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2,
          int* const __restrict__ color, int* const __restrict__ wl)
{
  const int lane = threadIdx.x % WS; // Lane within a warp
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  int maxrange = -1;  // Initialize the maximum neighbor range seen so far to -1.

  // Each thread processes multiple nodes
  for (int v = thread; __any_sync(Warp, v < nodes); v += threads) {  // Loop over vertices starting from your thread ID, moving by total number of threads.
    /* __any_sync(Warp, v < nodes): makes sure at least one thread in the warp is still working on valid vertex v. */

    bool cond = false;
    int beg, end, pos, degv, active;

    // For valid nodes
    if (v < nodes) {
      beg = nidx[v];
      end = nidx[v + 1];
      degv = end - beg; // Degree of node
      cond = (degv >= WS); // If degree is high enough, mark for large kernel

      if (cond) {
        wl[atomicAdd(&wlsize, 1)] = v; // Add to worklist
      } else {
        active = 0;
        pos = beg;
        // Check priority rules for neighbors
        for (int i = beg; i < end; i++) {
          const int nei = nlist[i];
          const int degn = nidx[nei + 1] - nidx[nei];
          if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
            active |= (unsigned int)MSB >> (i - beg);
            pos++;
          }     /*Priority Condition: Vertex v has higher priority if its degree is less or, for equal degrees, it has smaller hash or smaller id.
                If v has priority:
                Set corresponding bit in active bitmap (MSB is probably a bitmask like 0x80000000).
                Move pos forward. */
        }
      }
    }

    // Process nodes with large degrees using warp ballot
    int bal = __ballot_sync(Warp, cond);
    while (bal != 0) {
      const int who = __ffs(bal) - 1;   //  find first set bit (lowest thread with active vertex).
      bal &= bal - 1; // Clear that bit from bal.
      // Broadcast v, beg, end, degree from thread who to the entire warp.
      const int wv = __shfl_sync(Warp, v, who);
      const int wbeg = __shfl_sync(Warp, beg, who);
      const int wend = __shfl_sync(Warp, end, who);
      const int wdegv = wend - wbeg;
      int wpos = wbeg;  // Working position pointer.

      // Rebuild nlist2 with priority neighbors
      for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
        int wnei;
        bool prio = false;
        if (i < wend) {
          wnei = nlist[i];
          const int wdegn = nidx[wnei + 1] - nidx[wnei];
          prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (hash(wv) < hash(wnei))) || ((wdegv == wdegn) && (hash(wv) == hash(wnei)) && (wv < wnei)));
        }
        const int b = __ballot_sync(Warp, prio);
        const int offs = __popc(b & ((1 << lane) - 1)); //  how many set bits before me in the warp.
        if (prio) nlist2[wpos + offs] = wnei;
        wpos += __popc(b);
      }
      if (who == lane) pos = wpos;
    }

    if (v < nodes) {
      const int range = pos - beg; // Calculate how many active neighbors it had.
      maxrange = max(maxrange, range); 
      color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;  // If vertex is big-degree or has no active neighbors, store range  else active bitmap
      posscol[v] = (range >= WS) ? -1 : (MSB >> range); // possible colors. If too many neighbors, set to -1. Otherwise set bitmask.
    }
  }

  if (maxrange >= Mask) { printf("too many active neighbors\n"); asm("trap;"); }

  // Initialize secondary color possibilities
  for (int i = thread; i < edges / WS + 1; i += threads)
    posscol2[i] = -1;
}


============================================
========== Run Kernels for Large Degree Vertices ==========
============================================

// Kernel to process nodes with many neighbors (heavy work)
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runLarge(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ posscol, int* const __restrict__ posscol2, volatile int* const __restrict__ color, const int* const __restrict__ wl)
{
  const int stop = wlsize;
  if (stop != 0) {
    const int lane = threadIdx.x % WS;
    const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int threads = gridDim.x * ThreadsPerBlock;
    bool again;
    do {
      again = false;
      for (int w = thread; __any_sync(Warp, w < stop); w += threads) {
        // __any_sync: warp-wide check to see if any threads still have work.

        bool shortcut, done, cond = false;
        int v, data, range, beg, pcol;
        if (w < stop) {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0) {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }
        // Create a bitmask across the warp: which threads have a valid vertex to process.
        int bal = __ballot_sync(Warp, cond);
        while (bal != 0) {
          const int who = __ffs(bal) - 1;
          bal &= bal - 1; // Identify the first thread with work (who) and clear its bit.
          // Broadcast everything about the selected thread’s vertex to the rest of the warp.

          const int wdata = __shfl_sync(Warp, data, who);
          const int wrange = wdata >> (WS / 2);
          const int wbeg = __shfl_sync(Warp, beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const int wend = wbeg + wmaxcol;
          const int woffs = wbeg / WS;
          int wpcol = __shfl_sync(Warp, pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
            int nei, neidata, neirange;
            if (i < wend) {
              nei = nlist[i];
              neidata = color[nei];
              neirange = neidata >> (WS / 2);
              const bool neidone = (neirange == 0);  // Check if the neighbor has been finalized (neirange == 0).
              wdone &= neidone; //consolidated below

              // If Neighbor Is Colored: Remove its color from wpcol or Atomically remove it from posscol2 if in same range.
              if (neidone) {
                const int neicol = neidata;
                if (neicol < WS) {
                  wpcol &= ~((unsigned int)MSB >> neicol); //consolidated below
                } else {  
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0)) {
                    atomicAnd((int*)&posscol2[woffs + neicol / WS], ~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              } else {  // If neighbor is still active and overlaps with our color range, shortcut fails.
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; //consolidated below
              }
            }
          }
          wshortcut = __all_sync(Warp, wshortcut);   // Check if all threads in the warp agree on shortcut and done.
          wdone = __all_sync(Warp, wdone);
          Efficient bitwise AND reduction across all threads in warp to get common possible colors.
          wpcol &= __shfl_xor_sync(Warp, wpcol, 1);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 2);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 4);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 8);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 16);
          // Only the owner thread writes back the warp-computed values.
          if (who == lane) pcol = wpcol;
          if (who == lane) done = wdone;
          if (who == lane) shortcut = wshortcut;
        }

        if (w < stop) {
          if (range > 0) {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0) {
              const int offs = beg / WS;
              mc = max(1, mincol / WS);
              while ((val = posscol2[offs + mc]) == 0) mc++;
            }
            int newmincol = mc * WS + __clz(val);  // __clz to find leading (leftmost) color index.
            if (mincol != newmincol) shortcut = false;
            if (shortcut || done) {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            } else {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (__any_sync(Warp, again));  // Repeat the whole loop if any thread wants another iteration (due to changed color range).
  }
}

============================================
========== Run Kernels for Small Degree Vertices ==========
============================================

// Kernel to process nodes with few neighbors (light work)

// __restrict__: no pointer aliasing — better compiler optimizations.



static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runSmall(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, volatile int* const __restrict__ posscol, int* const __restrict__ color)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  if (thread == 0) wlsize = 0;  // Only thread 0 resets wlsize.

  bool again;
  do {
    again = false;
    for (int v = thread; v < nodes; v += threads) {
      __syncthreads();  // optional
      int pcol = posscol[v];
      if (__popc(pcol) > 1) {   // Fetch possible colors bitmap for vertex v. If more than 1 color available, we need to further refine.
        const int beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do {
          const int old = active;
          active &= active - 1; 
          const int curr = old ^ active;  // Extract lowest set bit (one neighbor at a time).
          const int i = beg + __clz(curr); // __clz(curr): Count Leading Zeros → position of the current bit → compute neighbor index.
          const int nei = nlist[i];
          const int neipcol = posscol[nei]; // Fetch the neighbor vertex and its possible colors.
          allnei |= neipcol;
          if ((pcol & neipcol) == 0) {  // If no shared colors between v and nei:
            pcol &= pcol - 1;  // Remove lowest set bit from pcol.
            keep ^= curr; // Remove this neighbor from keep (stop considering it).
          } else if (__popc(neipcol) == 1) {   // If neighbor nei is already colored (only 1 color left): Remove its color from pcol. Remove neighbor from keep.
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);

        // Shortcuts: 
        // If keep still has some neighbors: Try shortcutting: pick the best color available that none of the neighbors can use. If successful, set only that color.
        if (keep != 0) {
          const int best = (unsigned int)MSB >> __clz(pcol);
          if ((best & ~allnei) != 0) {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0) keep = __clz(pcol);
        color[v] = keep;
        posscol[v] = pcol;
      }
    }
  } while (again);
}


============================================
========== GPUTimer Class ==========
============================================

// Simple CUDA event-based timer class for profiling
struct GPUTimer {
  cudaEvent_t beg, end;
  GPUTimer() { cudaEventCreate(&beg); cudaEventCreate(&end); }
  ~GPUTimer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
  void start() { cudaEventRecord(beg, 0); }
  float stop() { cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001f * ms; }
};


============================================
========== Main Program ==========
============================================

int main(int argc, char* argv[])
{
  printf("ECL-GC v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 2) { printf("USAGE: %s input_file_name\n\n", argv[0]); exit(-1); }
  if (WS != 32) { printf("ERROR: warp size must be 32\n\n"); exit(-1); }
  if (WS != sizeof(int) * 8) { printf("ERROR: bits per word must match warp size\n\n"); exit(-1); }
  if ((ThreadsPerBlock < WS) || ((ThreadsPerBlock % WS) != 0)) { printf("ERROR: threads per block must be a multiple of the warp size\n\n"); exit(-1); }
  if ((ThreadsPerBlock & (ThreadsPerBlock - 1)) != 0) { printf("ERROR: threads per block must be a power of two\n\n"); exit(-1); }

  // Read graph input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  int* const color = new int [g.nodes]; // Color array on host

  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) { printf("ERROR: there is no CUDA capable device\n\n"); exit(-1); }

  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;

  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
    deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  // Memory allocations on device
  int *nidx_d, *nlist_d, *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d;
  if (cudaMalloc((void**)&nidx_d, (g.nodes + 1) * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate nidx_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&nlist_d, g.edges * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate nlist_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&nlist2_d, g.edges * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate nlist2_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&posscol_d, g.nodes * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate posscol_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&posscol2_d, (g.edges / WS + 1) * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate posscol2_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&color_d, g.nodes * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate color_d\n\n"); exit(-1); }
  if (cudaMalloc((void**)&wl_d, g.nodes * sizeof(int)) != cudaSuccess) { printf("ERROR: could not allocate wl_d\n\n"); exit(-1); }

  // Copy data from host to device
  if (cudaMemcpy(nidx_d, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { printf("ERROR: copying nidx to device failed\n\n"); exit(-1); }
  if (cudaMemcpy(nlist_d, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { printf("ERROR: copying nlist to device failed\n\n"); exit(-1); }

  // Configure kernels
  const int blocks = SMs * mTpSM / ThreadsPerBlock;
  cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runLarge, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runSmall, cudaFuncCachePreferL1);
}


// (Note: the above handles setting up for kernel execution, timing, launch, and cleaning up. Actual kernel invocations & final validation are to be added further.)
