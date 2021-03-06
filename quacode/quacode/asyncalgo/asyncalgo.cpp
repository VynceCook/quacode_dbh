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

#include <iostream>
#include <quacode/asyncalgo.hh>

AsyncAlgo::AsyncAlgo() { }

AsyncAlgo::~AsyncAlgo() {
    for (const auto& m : mDomainsMutex)
        delete m;
}

void AsyncAlgo::run() {
    parallelTask();
    Gecode::Support::Event dummy;
    dummy.wait();
}

void AsyncAlgo::postedEq(const std::string&, int) {
    std::cerr << "Eq constraint is not implemented" << std::endl;
    GECODE_NEVER
}

void AsyncAlgo::postedAnd(bool, const std::string&, bool, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "And constraint is not implemented" << std::endl;
    GECODE_NEVER
}
void AsyncAlgo::postedOr(bool, const std::string&, bool, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "Or constraint is not implemented" << std::endl;
    GECODE_NEVER
}
void AsyncAlgo::postedImp(bool, const std::string&, bool, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "Imp constraint is not implemented" << std::endl;
    GECODE_NEVER
}
void AsyncAlgo::postedXOr(bool, const std::string&, bool, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "XOr constraint is not implemented" << std::endl;
    GECODE_NEVER
}

void AsyncAlgo::postedPlus(int, const std::string&, int, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "Plus constraint is not implemented" << std::endl;
    GECODE_NEVER
}
void AsyncAlgo::postedTimes(int, const std::string&, const std::string&, TComparisonType, const std::string&) {
    std::cerr << "Times constraint is not implemented" << std::endl;
    GECODE_NEVER
}
void AsyncAlgo::postedLinear(const std::vector<Monom>&, TComparisonType, const std::string&) {
    std::cerr << "Linear constraint is not implemented" << std::endl;
    GECODE_NEVER
}
