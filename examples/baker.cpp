/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *     Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *     Vincent Barichard, 2013
 *
 *  Last modified:
 *     $Date$ by $Author$
 *     $Revision$
 *
 *  This file is part of Quacode:
 *     http://quacode.barichard.com
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

#include <iostream>
#include <vector>

#include <quacode/qspaceinfo.hh>
#include <gecode/minimodel.hh>
#include <gecode/driver.hh>

#include <algorithms/logger.hh>
#include <algorithms/dumb.hh>
#include <algorithms/montecarlo.hh>
#include <algorithms/gen.hh>

using namespace Gecode;

#ifdef GECODE_HAS_GIST
namespace Gecode { namespace Driver {
    /// Specialization for QDFS
    template<typename S>
        class GistEngine<QDFS<S> > {
            public:
                static void explore(S* root, const Gist::Options& opt) {
                    (void) Gist::explore(root, false, opt);
                }
        };
}}
#endif

/**
 * \brief Options taking one additional parameter
 */
class BakerOptions : public Options {
public:
    /// Asynchronous algorithm which will cooperate with QuaCode
    AsyncAlgo *aAlgo;

    /// Print strategy or not
    Gecode::Driver::BoolOption _printStrategy;
    //Size of the domains
    Gecode::Driver::UnsignedIntOption _n;
    //Algorithm to use
    Gecode::Driver::UnsignedIntOption _algo;
    /// Initialize options for example with name \a s
    BakerOptions(const char* s, int n0)
        : Options(s),
        _printStrategy("-printStrategy","Print strategy",false),
        _n("-n", "Value used to restrict the domain of w1 in order to make the problem harder", n0),
        _algo("-algo", "Number of the algorithm to use : 0 - Logger, 1 - Montecarlo, 2 - GenAlgo, 3 - Dumb", 0)
        {
            add(_printStrategy);
            add(_n);
            add(_algo);
        }
    ///Destructor to free the algorithm pointer
    ~BakerOptions() {
        if (aAlgo) {
            delete aAlgo;
        }
    }
    /// Parse options from arguments \a argv (number is \a argc)
    void parse(int& argc, char* argv[]) {
        Options::parse(argc,argv);
    }
    /// Return true if the strategy must be printed
    bool printStrategy(void) const {
        return _printStrategy.value();
    }
    ///Return n
    unsigned int n() const {
        return _n.value();
    }
    ///Return the algorithm
    unsigned int algo() const {
        return _algo.value();
    }
    /// Print help message
    virtual void help(void) {
        Options::help();
    }
};

class QCSPBaker : public Script, public QSpaceInfo {
    IntVarArray X;

public:
    QCSPBaker(const BakerOptions& opt) : Script(opt), QSpaceInfo(*opt.aAlgo)
    {
        // DEBUT DESCRIPTION PB
        std::cout << "Loading problem" << std::endl;
        if (!opt.printStrategy()) strategyMethod(0); // disable build and print strategy

        using namespace Int;
        aAlgo.newVar(EXISTS,"w1",TYPE_INT,1,opt.n());
        aAlgo.newVar(EXISTS,"w2",TYPE_INT,1,opt.n());
        aAlgo.newVar(EXISTS,"w3",TYPE_INT,1,opt.n());
        aAlgo.newVar(EXISTS,"w4",TYPE_INT,1,opt.n());
        aAlgo.newVar(EXISTS,"w5",TYPE_INT,1,opt.n());
        aAlgo.newVar(FORALL,"f",TYPE_INT,1,opt.n());
        aAlgo.newVar(EXISTS,"c1",TYPE_INT,-1,1);
        aAlgo.newVar(EXISTS,"c2",TYPE_INT,-1,1);
        aAlgo.newVar(EXISTS,"c3",TYPE_INT,-1,1);
        aAlgo.newVar(EXISTS,"c4",TYPE_INT,-1,1);
        aAlgo.newVar(EXISTS,"c5",TYPE_INT,-1,1);
        aAlgo.newAuxVar("o1",TYPE_INT,-opt.n(),opt.n());
        aAlgo.newAuxVar("o2",TYPE_INT,-opt.n(),opt.n());
        aAlgo.newAuxVar("o3",TYPE_INT,-opt.n(),opt.n());
        aAlgo.newAuxVar("o4",TYPE_INT,-opt.n(),opt.n());
        aAlgo.newAuxVar("o5",TYPE_INT,-opt.n(),opt.n());

        IntVarArgs w(*this,5,1,opt.n());
        IntVar f(*this,1,opt.n());
        setForAll(*this, f);
        IntVarArgs c(*this,5,-1,1);
        IntVarArgs vaX;
        vaX << w << f << c;
        X = IntVarArray(*this, vaX);

        IntVar o1(*this,-opt.n(),opt.n()), o2(*this,-opt.n(),opt.n()), o3(*this,-opt.n(),opt.n()), o4(*this,-opt.n(),opt.n()), o5(*this,-opt.n(),opt.n());
        rel(*this, w[0] * c[0] == o1);
        rel(*this, w[1] * c[1] == o2);
        rel(*this, w[2] * c[2] == o3);
        rel(*this, w[3] * c[3] == o4);
        rel(*this, w[4] * c[4] == o5);
        rel(*this, o1 + o2 + o3 + o4 + o5 == f);

        aAlgo.postTimes(1,"w1","c1",CMP_EQ,"o1");
        aAlgo.postTimes(1,"w2","c2",CMP_EQ,"o2");
        aAlgo.postTimes(1,"w3","c3",CMP_EQ,"o3");
        aAlgo.postTimes(1,"w4","c4",CMP_EQ,"o4");
        aAlgo.postTimes(1,"w5","c5",CMP_EQ,"o5");
        std::vector<Monom> expr { {1,"o1"}, {1,"o2"}, {1,"o3"}, {1,"o4"}, {1,"o5"}};
        aAlgo.postLinear(expr,CMP_EQ,"f");

        //branch(*this, X, INT_VAR_NONE(), INT_VALUES_MIN());
        branch(*this, aAlgo, X, INT_VAR_NONE());

        // END OF PB DESCRIPTION
        aAlgo.closeModeling();
    }

    QCSPBaker(bool share, QCSPBaker& p) : Script(share,p), QSpaceInfo(*this,share,p)
    {
        X.update(*this,share,p.X);
    }

    virtual Space* copy(bool share) { return new QCSPBaker(share,*this); }

    void eventNewInstance(void) const {
        TScenario scenario;
        for (int i=0; i<X.size(); i++)
            scenario.push_back({ .min = X[i].varimp()->min(), .max = X[i].varimp()->max() });
        aAlgo.newPromisingScenario(scenario);
    }

    void print(std::ostream& os) const {
        strategyPrint(os);
    }
};

int main(int argc, char* argv[])
{
    BakerOptions opt("Baker Problem",40);
    opt.parse(argc,argv);

    switch (opt.algo()) {
        default:
        case 0:
            opt.aAlgo = new Logger();
            break;

        case 1:
            opt.aAlgo = new MonteCarlo();
            break;

        case 2:
            opt.aAlgo = new GenAlgo();
            break;

        case 3:
            opt.aAlgo = new DumbAlgorithm();
            break;
    }

    Script::run<QCSPBaker,QDFS,BakerOptions>(opt);

    return 0;
}
