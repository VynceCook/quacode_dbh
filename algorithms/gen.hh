/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *    Vincent Barichard, 2015
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

#ifndef __GEN_H__
#define __GEN_H__

#ifdef QUACODE_USE_CUDA

#include <quacode/asyncalgo.hh>
#include <algorithms/export.hh>
#include <cuda/constraints.hh>

#define GEN_MAX_VAR 512

class GenAlgo : public AsyncAlgo {
    /// Variables of the problem
    struct VarDesc {
        int idxInBinder;
        Gecode::TQuantifier q;
        std::string name;
        TVarType type;
        Interval dom;
        Interval curDom;
    };

    struct VarDescArray {
        int idxInBinder[GEN_MAX_VAR];
        Gecode::TQuantifier q[GEN_MAX_VAR];
        std::string name[GEN_MAX_VAR];
        TVarType type[GEN_MAX_VAR];
        int dom[2 * GEN_MAX_VAR];
        int curDom[2 * GEN_MAX_VAR];
        int savedDom[2 * GEN_MAX_VAR];

        size_t  next;

        VarDescArray() : next(0) {}
        void push_back(const VarDesc & v);
    };

    /// Copy constructor set private to disable it.
    GenAlgo(const GenAlgo&);

    /// Flag to know if the main thread finished its work
    bool mbQuacodeThreadFinished;
    /// Mutex to block the destructor
    Gecode::Support::Mutex mDestructor;
    Gecode::Support::Mutex mDomaine;
    /// Stores the number of variables of the binder
    int     mNbVars;
    int     mNbBinderVars;
    int     mLastChoice;
    bool    mDomChanged;

    VarDescArray            mVars;
	std::vector<uintptr_t>  mCstrs;

    size_t  mPopulationSize;
    size_t  mNbEpoch;

    // Restaure the domain of var in the interval ]from, to]
    void restaureDomaines(int from, int to);

public:
    /// Main constructor
    ALGORITHM_EXPORT GenAlgo();

    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    ALGORITHM_EXPORT virtual void newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max);
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    ALGORITHM_EXPORT virtual void newAuxVarCreated(const std::string& name, TVarType t, int min, int max);

    /// Function called when a new choice (\a iVar = variable index in the binder,
    /// \a min and \a max are the lower and upper bounds of the value) during search
    ALGORITHM_EXPORT virtual void newChoice(int iVar, int min, int max);
    /// Function called when a new promising scenario is discovered during search
    ALGORITHM_EXPORT virtual void newPromisingScenario(const TScenario& instance);
    /// Function called when the search ends with a successfull strategy
    ALGORITHM_EXPORT virtual void strategyFound();
    /// Function called when a failure occured during search
    ALGORITHM_EXPORT virtual void newFailure();
    /// Function called when the search ends with a global failure, problem unfeasible
    ALGORITHM_EXPORT virtual void globalFailure();

    /// Function called when a new 'v0 == v2' constraint is posted
    ALGORITHM_EXPORT virtual void postedEq(const std::string& v0, int val);

    /// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    ALGORITHM_EXPORT virtual void postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    ALGORITHM_EXPORT virtual void postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    ALGORITHM_EXPORT virtual void postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    ALGORITHM_EXPORT virtual void postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);

    /// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
    ALGORITHM_EXPORT virtual void postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    ALGORITHM_EXPORT virtual void postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    ALGORITHM_EXPORT virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0);

    /// Function executed when the thread starts
    ALGORITHM_EXPORT virtual void parallelTask(void);

    // Main destructor
    ALGORITHM_EXPORT virtual ~GenAlgo();

private:
    size_t      findVar(const std::string & name);
};

#endif
#endif
