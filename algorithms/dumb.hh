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

#ifndef __DUMB_HH__
#define __DUMB_HH__

#include <quacode/asyncalgo.hh>

class DumbAlgorithm : public AsyncAlgo {
    /// Copy constructor set private to disable it.
    DumbAlgorithm(const DumbAlgorithm&);

    /// Stores the number of variables
    int mNbVars;
public:
    /// Main constructor, \a killThread is set to false
    /// if we want that the main thread (Quacode) waits for
    /// the end of the asynchronous working thread
    DumbAlgorithm(bool killThread = true);

    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and \a v its value
    virtual void newVarCreated(int idx, Gecode::TQuantifier q, std::string name, TVarType t, TVal v);
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and \a v its value
    virtual void newAuxVarCreated(std::string name, TVarType t, TVal v);

    /// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
    virtual void postedPlus(int n0, std::string v0, int n1, std::string v1, TComparisonType cmp, std::string v2);
    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    virtual void postedTimes(int n, std::string v0, std::string v1, TComparisonType cmp, std::string v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, std::string v0);

    /// Function called when a new choice (\a iVar = variable index in the binder, \a val is the value)
    /// during search
    virtual void newChoice(int iVar, TVal val);
    /// Function called when a new promising scenario is discovered during search
    virtual void newPromisingScenario(const TScenario& instance);
    /// Function called when the search ends with a successfull strategy
    virtual void strategyFound();
    /// Function called when a failure occured during search
    virtual void newFailure();
    /// Function called when the search ends with a global failure, problem unfeasible
    virtual void globalFailure();

    /// Function executed when the thread starts
    virtual void parallelTask(void);

    // Main destructor
    virtual ~DumbAlgorithm();
};

#endif