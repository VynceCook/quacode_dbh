/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main authors:
 *     Vincent Barichard <Vincent.Barichard@univ-angers.fr>
 *
 *  Copyright:
 *     Vincent Barichard, 2015
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
class RndQCSPOptions : public Options {
public:
    /// Asynchronous algorithm which will cooperate with QuaCode
    AsyncAlgo *aAlgo;
    /// Print strategy or not
    Gecode::Driver::BoolOption _printStrategy;
    /// File name of bench
    Gecode::Driver::StringValueOption _file;
    /// Initialize options for example with name \a s
    RndQCSPOptions(const char* s)
        : Options(s),
        _printStrategy("-printStrategy","Print strategy",false),
        _file("-file","File name of benchmark file")
    {
        add(_printStrategy);
        add(_file);
    }
    /// Return true if the strategy must be printed
    bool printStrategy(void) const {
        return _printStrategy.value();
    }
    /// Return file name
    const char *file(void) const {
        return _file.value();
    }
};

class RndQCSP : public Script, public QSpaceInfo {
    IntVarArray X;

public:
    RndQCSP(const RndQCSPOptions& opt) : Script(opt), QSpaceInfo(*opt.aAlgo) {
        std::cout << "Loading problem" << std::endl;
        using namespace Int;

        if (!opt.printStrategy()) strategyMethod(0); // disable build and print strategy
        if (!opt.file()) {
            throw Gecode::Exception("rndQCSP","Unable to open file");
        }

        IntVarArgs vaX;
        int nbVars=0;

        static const int READ_DESC = 0;
        static const int READ_VAR = 1;
        static const int READ_CST = 2;
        std::ifstream f(opt.file());
        std::string line;
        if (f.good()) {
            int state = READ_DESC;
            while ( !f.eof() ) {
                getline(f,line);
                if ( line.empty() ) break;
                std::istringstream iLine(line);
                int val;
                switch (state) {
                    case READ_DESC:
                        iLine >> nbVars; // Read number of variables
                        iLine >> val; // Read number of existential variables
                        iLine >> val; // Read number of universal variables
                        state = READ_VAR;
                        break;
                    case READ_VAR:
                        {
                            char quant;
                            int domMin, domMax;
                            iLine >> val; // Read idx variable
                            iLine >> quant; // Read quantifier
                            iLine >> domMin; // Read minimal value of domain
                            iLine >> domMax; // Read maximal value of domain
                            IntVar v(*this,domMin,domMax);
                            if (quant == 'F') setForAll(*this,v);
                            std::stringstream ss_x; ss_x << "x" << val;
                            if (quant == 'F')
                                aAlgo.newVar(FORALL,ss_x.str(),TYPE_INT,domMin,domMax);
                            else
                                aAlgo.newVar(EXISTS,ss_x.str(),TYPE_INT,domMin,domMax);
                            vaX << v;
                            nbVars--;
                            if (nbVars == 0) {
                                X = IntVarArray(*this, vaX);
                                state = READ_CST;
                            }
                        }
                        break;
                    case READ_CST:
                        {
                            int idx, coeff;
                            bool firstElt=true;
                            LinIntExpr liExp;
                            while (iLine >> coeff >> idx) {
                                std::cout << " + " << coeff << "*X_" << idx;
                                if (firstElt)
                                    liExp = LinIntExpr(X[idx],coeff);
                                else
                                    liExp = liExp + LinIntExpr(X[idx],coeff);
                                firstElt=false;
                            }
                            std::cout << std::endl;
                            rel(*this, liExp == 0);
                        }
                        break;
                }
            }
        }

        f.close();

        //branch(*this, X, INT_VAR_NONE(), INT_VALUES_MIN());
        branch(*this, X, INT_VAR_NONE(), INT_VAL_MIN());

        // END OF PB DESCRIPTION
        aAlgo.closeModeling();
    }

    RndQCSP(bool share, RndQCSP& p) : Script(share,p), QSpaceInfo(*this,share,p)
    {
        X.update(*this,share,p.X);
    }

    virtual Space* copy(bool share) { return new RndQCSP(share,*this); }

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
    RndQCSPOptions opt("RndQCSP Problem");
    opt.parse(argc,argv);

    Logger aAlgo;
    opt.aAlgo = &aAlgo;
    Script::run<RndQCSP,QDFS,RndQCSPOptions>(opt);

    return 0;
}
