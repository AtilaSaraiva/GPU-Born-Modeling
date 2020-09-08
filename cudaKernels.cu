__global__ void receptors(int it, int nr, int gxbeg, float *d_u1, float *d_data)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gx < nr){
        d_data[gx * c_nt + it] = d_u1[(gx + gxbeg + c_nb) * c_ny + c_nb];
    }
}

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it, int jsrc, int isrc)
{
    /*
    d_u             :pointer to an array on device where to add source term
    d_wavelet       :pointer to an array on device with source signature
    it              :time step id
    */
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    if ((gx == jsrc + c_nb) && (gy == isrc + c_nb))
    {
        d_u[idx] += d_wavelet[it];
    }
}

__device__ void set_halo(float *global, float shared[][SDIMX], int tx, int ty, int sx, int sy, int gx, int gy, int nx, int ny)
{
    /*
    global      :pointer to an array in global memory (gmem)
    shared      :2D array in shared device memory
    tx, ty      :thread id's in a block
    sx, sy      :thread id's in a shared memory tile
    gx, gy      :thread id's in the entire computational domain
    */

    // Each thread copies one value from gmem into smem
    shared[sy][sx] = global[gx * ny + gy];

    // Populate halo regions in smem for left, right, top and bottom boundaries of a block
    // if thread near LEFT border of a block
    if (tx < HALO)
    {
        // if global left
        if (gx < HALO)
        {
            // reflective boundary
            shared[sy][sx - HALO] = 0.0;
        }
        else
        {
            // if block left
            shared[sy][sx - HALO] = global[(gx - HALO) * ny + gy];
        }
    }
    // if thread near RIGHT border of a block
    if ((tx >= (BDIMX - HALO)) || ((gx + HALO) >= nx))
    {
        // if global right
        if ((gx + HALO) >= nx)
        {
            // reflective boundary
            shared[sy][sx + HALO] = 0.0;
        }
        else
        {
            // if block right
            shared[sy][sx + HALO] = global[(gx + HALO) * ny + gy];
        }
    }

    // if thread near BOTTOM border of a block
    if (ty < HALO)
    {
        // if global bottom
        if (gy < HALO)
        {
            // reflective boundary
            shared[sy - HALO][sx] = 0.0;
        }
        else
        {
            // if block bottom
            shared[sy - HALO][sx] = global[gx * ny + gy - HALO];
        }
    }

    // if thread near TOP border of a block
    if ((ty >= (BDIMY - HALO)) || ((gy + HALO) >= ny))
    {
        // if global top
        if ((gy + HALO) >= ny)
        {
            // reflective boundary
            shared[sy + HALO][sx] = 0.0;
        }
        else
        {
            // if block top
            shared[sy + HALO][sx] = global[gx * ny + gy + HALO];
        }
    }
}

// FD kernel
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp)
{
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_ny;

    // FD coefficient dt2 / dx2
    const float dt2dx2 = c_dt2dx2;

    // Thread address (ty, tx) in a block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // Thread address (sy, sx) in shared memory
    const unsigned int sx = threadIdx.x + HALO;
    const unsigned int sy = threadIdx.y + HALO;

    // Thread address (gy, gx) in global memory
    const unsigned int gx = blockIdx.x * blockDim.x + tx;
    const unsigned int gy = blockIdx.y * blockDim.y + ty;

    // Global linear index
    const unsigned int idx = gx * ny + gy;

    // Allocate shared memory for a block (smem)
    __shared__ float s_u1[SDIMY][SDIMX];
    __shared__ float s_u2[SDIMY][SDIMX];
    __shared__ float s_vp[SDIMY][SDIMX];

    // If thread points into the physical domain
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //       gmem, smem,  block, shared, global, dims
        set_halo(d_u1, s_u1, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_u2, s_u2, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        // Central point of fd stencil, o o o o x o o o o
        float du2_xx = c_coef[0] * s_u2[sy][sx];
        float du2_yy = c_coef[0] * s_u2[sy][sx];

#pragma unroll
        for (int d = 1; d <= 4; d++)
        {
            du2_xx += c_coef[d] * (s_u2[sy][sx - d] + s_u2[sy][sx + d]);
            du2_yy += c_coef[d] * (s_u2[sy - d][sx] + s_u2[sy + d][sx]);
        }
        // Second order wave equation
        d_u1[idx] = 2.0 * s_u2[sy][sx] - s_u1[sy][sx] + s_vp[sy][sx] * s_vp[sy][sx] * (du2_xx + du2_yy) * dt2dx2;

        __syncthreads();
    }
}
