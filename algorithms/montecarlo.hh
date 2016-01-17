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

#ifndef __MONTECARLO_HH__
#define __MONTECARLO_HH__

#include <vector>
#include <string>
#include <quacode/asyncalgo.hh>
#include <algorithms/export.hh>

class MonteCarlo : public AsyncAlgo {
    /// Constants
    constexpr static float sTemperatureDecreaseRate = 0.98;

    /// Flag to know if the main thread finished its work
    bool mbQuacodeThreadFinished;
    /// Mutex to block the destructor
    Gecode::Support::Mutex mDestructor;

    /// Copy constructor set private to disable it.
    MonteCarlo(const MonteCarlo&);

    /// Variables of the problem
    struct VarDesc {
        int idxInBinder;
        Gecode::TQuantifier q;
        std::string name;
        TVarType type;
        Interval dom;
    };
    /// Stores the number of variables
    int mNbVars;
    /// Number of variables of the binder
    int mNbBinderVars;
    /// Vector of variables
    std::vector< VarDesc > mVars;

    /// Constraints of the problem
    struct MX {
        int coeff;
        int iVar;
    };
    typedef std::vector< MX > TConstraint;;
    std::vector< TConstraint > mLinearConstraints;
    std::vector< TConstraint > mTimesConstraints;

    /// Array of conflicts evaluations
    std::vector< std::vector<int> > mConflicts;

    /// Function that returns the index of the variable
    /// \a name in our data structure. It returns -1 if the variable
    /// is undefined
    int getIdxVar(const std::string& name) const;

    /// Eval all constraints of the problem and update the number of conflict variables
    /// with the given instance \a instance. It returns the total error of the constraints.
    unsigned long int evalConstraints(const std::vector<int>& instance);

    /// Returns true if Metropolis rule is satisfied (it updates the temp variable)
    bool metropolis(int delta, float& temp);

public:
    /// Main constructor
    ALGORITHM_EXPORT MonteCarlo();

    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    ALGORITHM_EXPORT virtual void newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max);
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    ALGORITHM_EXPORT virtual void newAuxVarCreated(const std::string& name, TVarType t, int min, int max);

    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    ALGORITHM_EXPORT virtual void postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    ALGORITHM_EXPORT virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0);

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

    /// Function executed when the thread starts
    ALGORITHM_EXPORT virtual void parallelTask(void);

    // Main destructor
    ALGORITHM_EXPORT virtual ~MonteCarlo();
};

#endif
