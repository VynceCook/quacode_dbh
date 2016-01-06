#include <cuda/constraints.hh>
#include <cuda/kernels.hh>
#include <cuda/helper.hh>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>

#define CSTR_VAL_2X     CSTR_NO,  CSTR_NO
#define CSTR_VAL_4X     CSTR_VAL_2X,  CSTR_VAL_2X
#define CSTR_VAL_8X     CSTR_VAL_4X,  CSTR_VAL_4X
#define CSTR_VAL_16X    CSTR_VAL_8X,  CSTR_VAL_8X
#define CSTR_VAL_32X    CSTR_VAL_16X, CSTR_VAL_16X
#define CSTR_VAL_64X    CSTR_VAL_32X, CSTR_VAL_32X
#define CSTR_VAL_128X   CSTR_VAL_64X, CSTR_VAL_64X
#define CSTR_VAL_256X   CSTR_VAL_128X, CSTR_VAL_128X
#define CSTR_VAL_512X   CSTR_VAL_256X, CSTR_VAL_256X


CUDA_DEVICE __constant__ uintptr_t  cstrData[CSTR_MAX_CSTR * 8] = {CSTR_VAL_512X, CSTR_VAL_512X};
CUDA_DEVICE __constant__ TVarType   cstrType[CSTR_MAX_VAR];
CUDA_DEVICE __constant__ Gecode::TQuantifier cstrQuan[CSTR_MAX_VAR];
CUDA_DEVICE __constant__ int        cstrDom[CSTR_MAX_VAR];
CUDA_DEVICE __constant__ size_t     cstrPoly[CSTR_MAX_POLY];
                         size_t     cstrPolyNext = 0;
                         size_t     cstrVarNumber = 0;
                         size_t     cstrDomSize = 0;

CUDA_DEVICE              curandState_t *cstrRandStates = nullptr;

CUDA_DEVICE cstrFuncPtr     cstrTable[64] = {
        &cstrEq,       NULL,          NULL,          NULL,
        NULL,          NULL,          NULL,          NULL,

        &cstrAndNQ,    &cstrAndEQ,    &cstrAndLQ,    &cstrAndLE,
        &cstrAndGQ,    &cstrAndGR,    NULL,          NULL,

        &cstrOrNQ,     &cstrOrEQ,     &cstrOrLQ,     &cstrOrLE,
        &cstrOrGQ,     &cstrOrGR,     NULL,          NULL,

        &cstrImpNQ,    &cstrImpEQ,    &cstrImpLQ,    &cstrImpLE,
        &cstrImpGQ,    &cstrImpGR,    NULL,          NULL,

        &cstrXorNQ,    &cstrXorEQ,    &cstrXorLQ,    &cstrXorLE,
        &cstrXorGQ,    &cstrXorGR,    NULL,          NULL,

        &cstrPlusNQ,   &cstrPlusEQ,   &cstrPlusLQ,   &cstrPlusLE,
        &cstrPlusGQ,   &cstrPlusGR,   NULL,          NULL,

        &cstrTimesNQ,  &cstrTimesEQ,  &cstrTimesLQ,  &cstrTimesLE,
        &cstrTimesGQ,  &cstrTimesGR,  NULL,          NULL,

        &cstrLinearNQ, &cstrLinearEQ, &cstrLinearLQ, &cstrLinearLE,
        &cstrLinearGQ, &cstrLinearGR, NULL,          NULL
};

CUDA_HOST   size_t pushPolyToGPU(size_t * poly, size_t size) {
    size_t next = cstrPolyNext;
    assert(next + 2 * size < CSTR_MAX_POLY);

    CCR(cudaMemcpyToSymbol(cstrPoly, poly, size * sizeof(size_t), next * sizeof(size_t)));
    cstrPolyNext += 2 * size;

    return next;
}

CUDA_HOST   void pushVarToGPU(TVarType * type, Gecode::TQuantifier * quant, size_t size) {
    assert(size < CSTR_MAX_VAR);
    assert(type != nullptr);
    assert(quant != nullptr);

    CCR(cudaMemcpyToSymbol(cstrType, type, size * sizeof(TVarType)));
    CCR(cudaMemcpyToSymbol(cstrQuan, quant, size * sizeof(Gecode::TQuantifier)));

    cstrVarNumber = size;
}

CUDA_HOST void pushDomToGPU(int * dom, size_t size) {
    assert(size < CSTR_MAX_VAR);
    assert(size == 2 * cstrVarNumber);
    assert(dom != nullptr);

    CCR(cudaMemcpyToSymbol(cstrDom, dom, size * sizeof(int)));

    cstrDomSize = 0;

    for (size_t i = 0; i < size; i += 2) {
        cstrDomSize += (dom[i + 1] - dom[i]);
    }
}

CUDA_HOST void pushCstrToGPU(uintptr_t * cstrs, size_t size) {
    assert(size < (CSTR_MAX_CSTR * 8));
    assert(cstrs != nullptr);

    CCR(cudaMemcpyToSymbol(cstrData, cstrs, size * sizeof(uintptr_t)));
}

/**
 * Calls the kernel which initializes the population
 * @param popSize number of individuals in the population
 * @param indSize number of variables in an individual
 * @return initialized population
 */
CUDA_HOST   int *   initPopulation(size_t popSize, size_t indSize) {
    dim3 grid, block;
    int * d_pop;
    curandState_t *state;

    block = dim3(BLOCK_SIZE);
    grid = dim3((popSize + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    CCR(cudaMalloc((void**)&d_pop, sizeof(int) * popSize * indSize));
    CCR(cudaMalloc((void**)&state, sizeof(curandState_t) * popSize));
    CCR(cudaMemcpyToSymbol(cstrRandStates, &state, sizeof(curandState_t*)));
    initPopulationKernel<<<grid, block>>>(d_pop, popSize, indSize);
    CCR(cudaGetLastError());

    return d_pop;
}

CUDA_HOST   void    doTheMagic(int * pop, size_t popSize, size_t indSize, size_t gen) {
    dim3 grid, block;

    assert(pop != nullptr);

    //TODO set block and grid size more effectivelly
    block = dim3(BLOCK_SIZE);
    grid = dim3((popSize + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    doTheMagicKernel<<<grid, block>>>(pop, popSize, indSize, gen);
    CCR(cudaGetLastError());
}

CUDA_HOST   size_t*    getResults(int * pop, size_t popSize, size_t indSize, size_t * resSize) {
    dim3 grid, block;
    static size_t * d_res=  nullptr;
    size_t * h_res = nullptr;
    static size_t domSize = 0;

    // TODO set block & grid size
    block = dim3(BLOCK_SIZE);
    grid = dim3((domSize + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    assert(pop != nullptr);

    if (domSize == 0 || cstrDomSize != domSize) {
        domSize = cstrDomSize;

        if (d_res) {
            CCR(cudaFree((void*)d_res));
        }

        CCR(cudaMalloc((void**)&d_res, sizeof(size_t) * domSize));
    }

    getResultsKernel<<<256, 32>>>(pop, popSize, indSize, cstrDomSize, d_res);
    CCR(cudaGetLastError());
    CCR(cudaFree((void*)pop));

    h_res = new size_t[cstrDomSize];
    *resSize = cstrDomSize;

    CCR(cudaMemcpy(h_res, d_res, sizeof(size_t) * cstrDomSize, cudaMemcpyDeviceToHost));

    return h_res;
}

/**
 * Randomly creates a population of candidates to evolve
 * @param popPtr the address where the population will be stored
 * @param popSize number of individuals
 * @param indSize size of an an individual
 */
CUDA_GLOBAL void    initPopulationKernel(int * popPtr, size_t popSize, size_t indSize) {
    size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(CURAND_SEED, gtid, 0, &cstrRandStates[gtid]);

    if (gtid < popSize){
        for (int i = 0; i<indSize; ++i){
            popPtr[indSize * gtid + i] = CurandInterval(curand(&cstrRandStates[gtid]), cstrDom[2 * i], cstrDom[(2 * i) + 1]);
            // Variable i is in [cstrDom[2i], cstrDom[2i + 1]]
        }
    }
}

/**
 * "Evolves" an individual (a set of values) to give it the lowest score
 * possible (We are looking for the worst possible candidat, to give the solver
 * some hint, where he must not search
 * @param pop the candidate population
 * @param popSize how many individual are in this population
 * @param indSize the individual's size (ints)
 * @param gen number of generations (epochs) before stopping
 */
CUDA_GLOBAL void    doTheMagicKernel(int * pop, size_t popSize, size_t indSize, size_t gen) {
    size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int old_fitness, cur_fitness;
    int * indiv = pop + (gtid * indSize); // points at the first element of our current individual
    int * child = new int[indSize];       // candidate for the next generation
    int mut_var = 0;                      // Mutated variable

    old_fitness = cstrValidate(indiv);

    for (int i = 0;  i < indSize; ++i){
        child[i] = indiv[i];
    }

    if (gtid < popSize){
        for (int epoch = 0; epoch < gen && old_fitness > 0; ++epoch){
            mut_var = CurandInterval(curand(&cstrRandStates[gtid]), 0, indSize - 1);
            child[mut_var] = CurandInterval(curand(&cstrRandStates[gtid]), cstrDom[2 * mut_var], cstrDom[(2 * mut_var) + 1]);
            cur_fitness = cstrValidate(child);
            if (cur_fitness < old_fitness){
                // We save the child
                indiv[mut_var] = child[mut_var];
                old_fitness = cur_fitness;
            }
            else{
                // We reset the child
                child[mut_var] = indiv[mut_var];
            }
        }
    }
}

/**
 * Counts how many occurences of a specific value we have, for a given variable
 * @param pop candidates population
 * @param popSize number of individuals in the population
 * @param indSize individual size (how many ints)
 * @param domSize the sum of each constraint's domain size
 * @param res number of occurencies of each variable's value
 */
CUDA_GLOBAL void    getResultsKernel(int * pop, size_t popSize, size_t indSize, size_t domSize, size_t* res) {
    size_t  gtid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t  sum = 0;
    size_t  idx = 0;          // variable's index
    int     val = cstrDom[0]; // value to test

    if (gtid < domSize) {
        // set val to the value we want to test
        for (size_t i = 0; (i < domSize) && (i < gtid); ++i) {
            ++val;

            if (val > cstrDom[2 * idx + 1]) {
                ++idx;
                val = cstrDom[2 * idx];
            }
        }

        for (size_t i = 0; i < popSize; ++i) {
            sum += (pop[i * indSize + idx] == val);
        }
        res[gtid] = sum;
    }
}

/**
 * Test each constraint on a candidate to evaluate it
 * @param c the candidate
 * @return how many constraints are satisfied
 */
CUDA_DEVICE int cstrValidate(int * c) {
    int satisfied = 0;
    for (size_t i = 0; cstrData[8 * i] != CSTR_NO && (8 * i) < CSTR_MAX_CSTR; ++i) {
        if (cstrTable[cstrData[8 * i]](cstrData + (8 * i) + 1, c)) {
            satisfied ++;
        }
    }

    return(satisfied);
}

CUDA_DEVICE bool cstrEq(uintptr_t * data, int * c) {
    size_t v0 = (size_t) data[0];
    int    val = uint2int((unsigned int) data[1]);

    return c[v0] == val;
}

CUDA_DEVICE bool cstrAndEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrAndNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrAndGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrAndGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrAndLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrAndLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpAnd(p0, c[v0], p1, c[v1]) <= c[v2];
}


CUDA_DEVICE bool cstrOrEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrOrNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrOrGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrOrGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrOrLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrOrLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpOr(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrImpEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrImpNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrImpGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrImpGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrImpLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrImpLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpImp(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrXorEQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrXorNQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrXorGQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrXorGR(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrXorLQ(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrXorLE(uintptr_t * data, int * c) {
    bool p0 = (bool) data[0], p1 = (bool) data[2];
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpXor(p0, c[v0], p1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrPlusEQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrPlusNQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrPlusGQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrPlusGR(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrPlusLQ(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrPlusLE(uintptr_t * data, int * c) {
    int n0 = uint2int((unsigned int) data[0]), n1 = uint2int((unsigned int) data[2]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[3], v2 = (size_t) data[4];

    return OpPlus(n0, c[v0], n1, c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrTimesEQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) == c[v2];
}

CUDA_DEVICE bool cstrTimesNQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) != c[v2];
}

CUDA_DEVICE bool cstrTimesGQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) > c[v2];
}

CUDA_DEVICE bool cstrTimesGR(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) >= c[v2];
}

CUDA_DEVICE bool cstrTimesLQ(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) < c[v2];
}

CUDA_DEVICE bool cstrTimesLE(uintptr_t * data, int * c) {
    int n = uint2int((unsigned int) data[0]);
    size_t v0 = (size_t) data[1], v1 = (size_t) data[2], v2 = (size_t) data[3];

    return OpTimes(n, c[v0], c[v1]) <= c[v2];
}

CUDA_DEVICE bool cstrLinearEQ(uintptr_t * data, int * c) {
    size_t vIdx = (size_t) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    size_t *v = cstrPoly + vIdx;
    int sum = 0;

    OpLinear(v, size, sum);
    return sum == c[v0];
}

CUDA_DEVICE bool cstrLinearNQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    OpLinear(v, size, sum);
    return sum != c[v0];
}

CUDA_DEVICE bool cstrLinearGQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    OpLinear(v, size, sum);
    return sum > c[v0];
}

CUDA_DEVICE bool cstrLinearGR(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    OpLinear(v, size, sum);
    return sum >= c[v0];
}

CUDA_DEVICE bool cstrLinearLQ(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    OpLinear(v, size, sum);
    return sum < c[v0];
}

CUDA_DEVICE bool cstrLinearLE(uintptr_t * data, int * c) {
    size_t * v = (size_t*) data[0], size = (size_t) data[1];
    size_t v0 = (size_t) data[2];
    int sum = 0;

    OpLinear(v, size, sum);
    return sum <= c[v0];
}
