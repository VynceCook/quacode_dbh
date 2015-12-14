/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *    Vincent Barichard, 2013
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */
#ifdef QUACODE_USE_CUDA
#include <cstdlib>
#include <ctime>
#include <algorithms/gen.hh>
#include <cuda/kernels.hh>
#define OSTREAM std::cerr
#define MAX_VARS_IN_INTERVAL 24
#define MUTATION_THRESHOLD 1.0

// #define DEBUG

#ifdef DEBUG
#define LOG(__s) {__s}

std::string printComp(TComparisonType cmp) {
    switch (cmp) {
        case CMP_NQ:
            return "!=";
        case CMP_EQ:
            return "==";
        case CMP_LQ:
            return "<";
        case CMP_LE:
            return "<=";
        case CMP_GQ:
            return ">";
        case CMP_GR:
            return ">=";
    }
    return "NaC";
}

std::string printType(TVarType t) {
    return (t ? "Int" : "Bool");
}
#else
#define LOG(__s) /* void */
#endif

Vector<size_t> mCstrs;

void GenAlgo::restaureDomaines(int from, int to) {
    for (int i = from + 1; i <= to; ++i) {
        mVars[i].curDom = mVars[i].dom;
    }
}

GenAlgo::GenAlgo() : AsyncAlgo() {
    mNbVars = 0;
    mNbBinderVars = 0;
    mLastChoice = 0;
}

GenAlgo::~GenAlgo() {
    mbQuacodeThreadFinished = true;

#ifndef QUACODE_USE_CUDA
    for (auto it = mCstrs.begin(); it != mCstrs.end(); ++it) {
        delete *it;
    }
#endif

    mDestructor.acquire();
    mDestructor.release();
}

void GenAlgo::newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New var " << idx << "/" <<  printType(t) << " : " << name << ", type " << (q == EXISTS ? "E" : "F") << ", dom {" << min << "," << max << "}" <<std::endl;)
    mVars.push_back({idx, q, name, t, min, max, min, max});
    ++mNbVars;
    ++mNbBinderVars;
}

void GenAlgo::newAuxVarCreated(const std::string& name, TVarType t, int min, int max) {
    LOG(OSTREAM << "New auxiliary var " << name << "/" << printType(t) << ", dom {" << min << "," << max << "}" << std::endl;)
    mVars.push_back({-1, EXISTS, name, t, min, max, min, max});
    ++mNbVars;
}

void GenAlgo::newChoice(int iVar, int min, int max) {
    LOG(OSTREAM << "Chef ! We need to explore others choices : " << iVar << " {" << min << "," << max << "}" << std::endl;)

    if (iVar < mLastChoice) {
        restaureDomaines(iVar, mLastChoice);
    }

    mVars[iVar].curDom = {min, max};
    mLastChoice = iVar;

    LOG(
    for(auto s : mVars) {
        OSTREAM << "{" << s.curDom.min << "," << s.curDom.max << "}" << ", ";
    }
    OSTREAM << std::endl;)
}

void GenAlgo::newPromisingScenario(const TScenario& scenario) {
    LOG(
    OSTREAM << "Chef ! I think this scenario is interesting : ";
    for (auto s: scenario) {
        OSTREAM << "{" << s.min << "," << s.max << "}, ";
    }
    OSTREAM << std::endl;
    )

    if (scenario.size()) return;
}

void GenAlgo::strategyFound() {
    LOG(OSTREAM << "Chef ! We just found a solution !" << std::endl;)
}

void GenAlgo::newFailure() {
    LOG(OSTREAM << "Another faillure chef !" << std::endl;)
}

void GenAlgo::globalFailure() {
    LOG(OSTREAM << "It can't be done. ABORT ! ABORT ! ABORT !" << std::endl;)
}

/// Function called when a new 'v0 == v2' constraint is posted
void GenAlgo::postedEq(const std::string& v0, int val) {
    LOG(OSTREAM << "New constraint Eq " << v0 << "=" << val << std::endl;)
    size_t v0Idx = findVar(v0);

    if (v0Idx != (size_t)-1) {
		mCstrs.insert(mCstrs.end(), {(CSTR_EQ_IDX << 3), v0Idx, val});
    }
    else {
        OSTREAM << "Can't find " << v0 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint And " << p0 << ":" << v0 << " && " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrBool::create(p0, v0Idx, OP_AND, p1, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_AND_IDX << 3) | cmp), p0, v0Idx, p1, v1Idx, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Or " << p0 << ":" << v0 << " || " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrBool::create(p0, v0Idx, OP_OR, p1, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_OR_IDX << 3) | cmp, p0, v0Idx, p1, v1Idx, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Imp " << p0 << ":" << v0 << " >> " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrBool::create(p0, v0Idx, OP_IMP, p1, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_IMP_IDX << 3) | cmp, p0, v0Idx, p1, v1Idx, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
void GenAlgo::postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Xor " << p0 << ":" << v0 << " ^ " << p1 << ":" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrBool::create(p0, v0Idx, OP_XOR, p1, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_XOR_IDX << 3) | cmp, p0, v0Idx, p1, v1Idx, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}


/// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
void GenAlgo::postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Plus " << n0 << "*" << v0 << " + " << n1 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrPlus::create(n0, v0Idx, n1, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_PLUS_IDX << 3) | cmp, n0, v0Idx, n1, v1Idx, v2Idx});
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
void GenAlgo::postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2) {
    LOG(OSTREAM << "New constraint Times " << n << "*" << v0 << "*" << v1 << " " << printComp(cmp) << " " << v2 << std::endl;)
    size_t v0Idx = findVar(v0), v1Idx = findVar(v1), v2Idx = findVar(v2);

    if ((v0Idx != (size_t)-1) && (v1Idx != (size_t)-1) && (v2Idx != (size_t)-1)) {
        // Constraint * tmp = CstrTimes::create(n, v0Idx, v1Idx, cmp, v2Idx);
        // mCstrs.push_back(tmp);
		mCstrs.insert(mCstrs.end(), {(CSTR_TIMES_IDX << 3) | cmp, n, v0Idx, v1Idx, v2Idx})
    }
    else {
        OSTREAM << "Can't find on of the variables " << v0 << ", " << v1 << ", " << v2 << std::endl;
        GECODE_NEVER
    }
}

/// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
void GenAlgo::postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0) {
    LOG(
    OSTREAM << "New constraint Linear ";
    for (auto m : poly) {
        OSTREAM << m.coeff << "*" << m.varName << " + ";
    }
    OSTREAM << printComp(cmp) << " " << v0 << std::endl;
    )

    size_t v0Idx = findVar(v0);

    if (v0Idx != (size_t)-1) {
        size_t * h_polyCpy = new size_t[poly.size() * 2], * h_polyCpyStart = h_polyCpy;
		size_t * d_polyCpy; // On the GPU

        for (auto it = poly.begin(); it != poly.end(); ++it) {
            size_t viIdx = findVar(it->varName);

            if (viIdx != (size_t)-1) {
                *h_polyCpy++ = it->coeff;
                *h_polyCpy++ = viIdx;
            }
            else {
                OSTREAM << "Can't find " << it->varName << std::endl;
                GECODE_NEVER
            }
        }
        // Constraint * tmp = CstrLinear::create(polyCpy2, poly.size(), cmp, v0Idx);
        // mCstrs.push_back(tmp);
		//
		// TODO Transfert array to GPU
		cudaMalloc(d_polyCpy, poly.size() * sizeof(size_t));
		cudaMemcpy(d_polyCpy, h_polyCpyStart, poly.size() * sizeof(size_t), cudaHostToDevice);
		mCstrs.insert(mCstrs.end(), {(CSTR_LINEAR_IDX << 3) | cmp, d_polyCpy, poly.size(), v0Idx);
        delete[] polyCpy2;
    }
    else {
        OSTREAM << "Can't find variable " << v0 << std::endl;
        GECODE_NEVER
    }
}


void GenAlgo::parallelTask() {
    LOG(OSTREAM << "THREAD start" << std::endl;)

    OSTREAM << "Calling foo kernel" << std::endl;
    foo();

    mDestructor.acquire();
    srand(time(NULL));
    for ( ; ; ) {
        if (mbQuacodeThreadFinished) break;

        Gecode::Support::Thread::sleep(300);
    }


    mDestructor.release();
    LOG(OSTREAM << "THREAD stop" << std::endl;)
}


size_t GenAlgo::findVar(const std::string & name) {
    for (size_t i = 0; i < mVars.size(); ++i) {
        if (mVars[i].name == name) {
            return i;
        }
    }

    return -1;
}

bool GenAlgo::evaluate(const std::vector<int> &vars) {
    return Constraint::evaluate(mCstrs.data(), mCstrs.size(), vars.data());
}

/*
// Seems useless, will be updated if needed

std::vector<int *> GenAlgo::generateAll() {
	std::vector<int *> res;
	int cur[mNbVars];
	int i;

	// Initializes all at minimal value
	for (i=0; i<mNbVars; ++i) {
		cur[i] = mVars[i].dom.min;
	}
	if (evaluate(cur)) {
		int * newArray = new int[mVars];
		memcpy(newArray, cur, mNbVars * sizeof(int));
		res.push_back(newArray);
	}

	// "Increases" the candidate
	while (1) {
		for (i=0; cur[i] == mVars[i].dom.max && i < mNbVars; ++i) {
			cur[i] = mVars[i].dom.min;
		}

		if (i == mNbVars) {
			// We generated all the possible var sets
			break;
		}
		cur[i] ++;

		if (evaluate(cur)) {
			int * newArray = new int[mVars];
			memcpy(newArray, cur, mNbVars * sizeof(int));
			res.push_back(newArray);
		}
	}

	return(res);
}


std::vector<std::vector<int>> GenAlgo::generateRandom(const int n) {
	std::vector<std::vector<int>> res(n);

	srand(time(NULL));

	// generates random sets of var (useful in genetic algo's initialisation)
	for (int i=0; i<n; ++i) {
		res[i] = std::vector<int>(mNbVars);
		for (int j=0; j<mNbVars; ++j) {
			res[i][j] = mVars[j].dom.min + (rand() % (mVars[j].dom.max - mVars[j].dom.min));
		}
	}

	return(res);
}
*/

void GenAlgo::execute(const int maxGen, const int popSize) {
	Interval **curGen;  // the current population, will be used as parents
	Interval **nextGen; // the next generation, will contain the children
	int range[popSize];
	int prevRange;
	float randP1, randP2;
	int parent1, parent2;

	curGen  = (Interval **)malloc(popSize * sizeof(Interval *));
	nextGen = (Interval **)malloc(popSize * sizeof(Interval *));
	for (int i=0; i<popSize; ++i){
		curGen[i]  = (Interval *)malloc(mNbVars * sizeof(Interval));
		nextGen[i] = (Interval *)malloc(mNbVars * sizeof(Interval));
	}

	// Candidates generation
	generateRandomPop(popSize, curGen);

	// for n gen
	for (int nGen=0; nGen<maxGen; ++nGen){
		prevRange = 0;
		// Candidates evaluation
		for (int i=0; i<popSize; ++i){
			range[i] = prevRange + evaluateIntervals(curGen[i]);
			prevRange += range[i];
		}

		// For each individual in the next generation...
		for (int i=0; i<popSize; ++i){
			// Breeding
			randP1 = randFloat(range[popSize - 1]);
			randP2 = randFloat(range[popSize - 1]);
			int j;
			for (j=0; randP1 > range[j]; ++j);
			parent1 = j;
			for (j=0; randP2 > range[j]; ++j);
			parent2 = j;
			crossover(curGen, nextGen, parent1, parent2, i);

			// Mutation
			if (randFloat(100) < MUTATION_THRESHOLD){
				mutate(nextGen, i);
			}
		}

		curGen = nextGen;
	}

	for (int i=0; i<popSize; ++i){
		free(curGen[i]);
		free(nextGen[i]);
	}
	free(curGen);
	free(nextGen);

}


void GenAlgo::generateRandomPop(const int size, Interval ** population){
	// Generate the base population
	for (int p=0; p<size; ++p){
		for (int i=0; i<mNbVars; ++i){
			population[p][i].min = mVars[i].dom.min + (rand() % (mVars[i].dom.max - mVars[i].dom.min));
			population[p][i].max = population[p][i].min + (rand() % (mVars[i].dom.max - population[p][i].min));
		}
	}
}


float GenAlgo::evaluateIntervals(const Interval * interVars) {
	// Evaluate an array of mNbVars intervals
	// Calculates the average of all the candidates' evaluations in the defined intervals
	int sum;
	int num = 1; // number of candidates in the intervals, at least 1
	std::vector <int> cur(mNbVars);
//	int cur[mNbVars];
	int i;

	// Initialize all to the minimal values
	for (i=0; i<mNbVars; ++i) {
		cur[mNbVars] = interVars[i].min;
	}

	sum = evaluate(cur);

	while (1) {
		// "Increase" the candidate
		for (i=0; cur[i] == interVars[i].max && i<mNbVars; ++i) {
			cur[i] = interVars[i].min;
		}

		if (i == mNbVars){
			break;
		}
		cur[i] ++;
		num ++;

		// Avoid some candidates with too many possibilities (would take too long)
		if (num > MAX_VARS_IN_INTERVAL)
			return(0);

		sum += evaluate(cur);
	}

	return(sum / num);
}


void GenAlgo::crossover(Interval ** parents, Interval ** children, int p1, int p2, int c){
	// Simple crossover
	int crossoverPoint = 1 + rand() % (mNbVars - 2);
	int i;

	for (i=0; i<crossoverPoint; ++i){
		children[c][i] = parents[p1][i];
	}
	for (; i<mNbVars; ++i){
		children[c][i] = parents[p2][i];
	}
}


void GenAlgo::mutate(Interval ** population, int p){
	// Randomly changes an interval in an individual
	int i = rand() % mNbVars;

	population[p][i].min = mVars[i].dom.min + (rand() % (mVars[i].dom.max - mVars[i].dom.min));
	population[p][i].max = population[p][i].min + (rand() % (mVars[i].dom.max - population[p][i].min));
}


float GenAlgo::randFloat(float max){
	return((float)rand()/(float)(RAND_MAX/max));
}

#endif
