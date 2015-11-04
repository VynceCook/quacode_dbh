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

#include <quacode/asyncalgo.hh>

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

    struct Constraint {
        typedef bool (*cmpFuncPtr)(int, int);
        typedef bool (*opFuncPtr)(bool, bool, bool, bool);

        virtual ~Constraint() {};
        virtual bool evaluate(const std::vector<int> &) = 0;

        static bool cmpEQ(int, int);
        static bool cmpNQ(int, int);
        static bool cmpGQ(int, int);
        static bool cmpGR(int, int);
        static bool cmpLQ(int, int);
        static bool cmpLE(int, int);
        static cmpFuncPtr getCmpPtr(TComparisonType);
    };

    // v0 == v1
    struct CstrEq: Constraint {
        size_t v0;
        int v2;

        CstrEq(size_t v0, int v2);
        virtual bool evaluate(const std::vector<int> &);
    };

    // p0v0 op p1v1 cmp v2
    struct CstrBool: Constraint {
        bool        p0;
        size_t      v0;
        opFuncPtr   op;
        bool        p1;
        size_t      v1;
        cmpFuncPtr  cmp;
        size_t      v2;

        CstrBool(bool p0, size_t v0, opFuncPtr op, bool p1, size_t v1, cmpFuncPtr cmp, size_t v2);
        virtual bool evaluate(const std::vector<int> &);

        static bool opAnd(bool, bool, bool, bool);
        static bool opOr(bool, bool, bool, bool);
        static bool opImp(bool, bool, bool, bool);
        static bool opXor(bool, bool, bool, bool);
    };

    // n0*v0 + n1*v1 cmp v2
    struct CstrPlus: Constraint {
        int n0;
        size_t v0;
        int n1;
        size_t v1;
        cmpFuncPtr cmp;
        size_t v2;

        CstrPlus(int n0, size_t v0, int n1, size_t v1, cmpFuncPtr cmp, size_t v2);
        virtual bool evaluate(const std::vector<int> &);
    };

    // n*v0*v1 cmp v2
    struct CstrTimes: Constraint {
        int n;
        size_t v0;
        size_t v1;
        cmpFuncPtr cmp;
        size_t v2;

        CstrTimes(int n, size_t v0, size_t v1, cmpFuncPtr cmp, size_t v2);
        virtual bool evaluate(const std::vector<int> &);
    };

    struct CstrLinear: Constraint {
        std::vector<std::pair<int, size_t>> poly;
        cmpFuncPtr cmp;
        size_t v0;

        ~CstrLinear() {}
        CstrLinear(const std::vector<std::pair<int, size_t>> &poly, cmpFuncPtr cmp, size_t v0);
        virtual bool evaluate(const std::vector<int> &);
    };

    /// Copy constructor set private to disable it.
    GenAlgo(const GenAlgo&);

    /// Flag to know if the main thread finished its work
    bool mbQuacodeThreadFinished;
    /// Mutex to block the destructor
    Gecode::Support::Mutex mDestructor;
    /// Stores the number of variables of the binder
    int mNbVars;
    int mNbBinderVars;
    int mLastChoice;

    std::vector<VarDesc> mVars;

    std::vector<Constraint*>  mCstrs;

    // Restaure the domain of var in the interval ]from, to]
    void restaureDomaines(int from, int to);

public:
    /// Main constructor
    GenAlgo();

    /// Function called when a new variable \a var named \a name
    /// is created at position \a idx in the binder.
    /// \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    virtual void newVarCreated(int idx, Gecode::TQuantifier q, const std::string& name, TVarType t, int min, int max);
    /// Function called when a new auxiliary  variable \a var named \a name
    /// is created. \a t is the type of the variable, and
    /// \a min and \a max are the lower and upper bounds of the domain
    virtual void newAuxVarCreated(const std::string& name, TVarType t, int min, int max);

    /// Function called when a new choice (\a iVar = variable index in the binder,
    /// \a min and \a max are the lower and upper bounds of the value) during search
    virtual void newChoice(int iVar, int min, int max);
    /// Function called when a new promising scenario is discovered during search
    virtual void newPromisingScenario(const TScenario& instance);
    /// Function called when the search ends with a successfull strategy
    virtual void strategyFound();
    /// Function called when a failure occured during search
    virtual void newFailure();
    /// Function called when the search ends with a global failure, problem unfeasible
    virtual void globalFailure();

    /// Function called when a new 'v0 == v2' constraint is posted
    virtual void postedEq(const std::string& v0, int val);

    /// Function called when a new 'p0v0 && p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedAnd(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 || p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 >> p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedImp(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'p0v0 ^ p1v1 <cmp> v2'  (p0, p1 are polarity of literals) constraint is posted
    virtual void postedXOr(bool p0, const std::string& v0, bool p1, const std::string& v1, TComparisonType cmp, const std::string& v2);

    /// Function called when a new 'n0*v0 + n1*v1 <cmp> v2' constraint is posted
    virtual void postedPlus(int n0, const std::string& v0, int n1, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'n*v0*v1 <cmp> v2' constraint is posted
    virtual void postedTimes(int n, const std::string& v0, const std::string& v1, TComparisonType cmp, const std::string& v2);
    /// Function called when a new 'SUM_i n_i*v_i <cmp> v0' constraint is posted
    virtual void postedLinear(const std::vector<Monom>& poly, TComparisonType cmp, const std::string& v0);

    /// Function executed when the thread starts
    virtual void parallelTask(void);

    // Main destructor
    virtual ~GenAlgo();

private:
    size_t      findVar(const std::string & name);
    bool        evaluate(const std::vector<int> & vars);
	std::vector<std::vector<int>> generateAll();
	std::vector<std::vector<int>> generateRandom(const int n);
};

#endif
