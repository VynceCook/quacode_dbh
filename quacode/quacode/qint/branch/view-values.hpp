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
#include <gecode/int/branch.hh>
#include <asyncalgo/asyncalgo.hh>
#define UNUSED_STREAM std::cout

namespace Gecode { namespace Int { namespace Branch {

  /**
   * \brief %QBrancher by view and values selection
   *
   */
  template<int n, bool min>
  class QViewValuesOrderBrancher : public ViewValuesBrancher<n,min> {
    typedef typename ViewBrancher<IntView,n>::BranchFilter BranchFilter;
  protected:
    using ViewBrancher<IntView,n>::x;
    using ViewBrancher<IntView,n>::start;
    /// Information about value and order to deal with this value
    class ValOrder {
    public:
      int v;
      unsigned int o;
      ValOrder(int _v, unsigned int _o) : v(_v), o(_o) {}
    };
    /// Offset of the brancher
    int offset;
    /// Asynchronous algorithm
    AsyncAlgo& aAlgo;
    /// Description of domains parse order of the variables involved in the brancher
//ICI    SharedArray< std::vector<ValOrder> > domainsParseOrder;
    /// Constructor for cloning \a b
    QViewValuesOrderBrancher(Space& home, bool shared, QViewValuesOrderBrancher& b);
    /// Constructor for creation
    QViewValuesOrderBrancher(Home home, AsyncAlgo& _aAlgo, ViewArray<IntView>& x, int _offset,
                             ViewSel<IntView>* vs[n], BranchFilter bf, IntVarValPrint vvp);
  public:
    /// Return choice
    virtual const Choice* choice(Space& home);
    /// Return choice
    virtual const Choice* choice(const Space& home, Archive& e);
    /// Perform commit for choice \a c and alternative \a a
    virtual ExecStatus commit(Space& home, const Choice& c, unsigned int a);
    /// Perform cloning
    virtual Actor* copy(Space& home, bool share);
    /// Delete brancher and return its size
    virtual size_t dispose(Space& home);
    /// Constructor for creation
    static BrancherHandle post(Home home, AsyncAlgo& aAlgo, ViewArray<IntView>& x, int offset,
                               ViewSel<IntView>* vs[n], BranchFilter bf, IntVarValPrint vvp);
  };

  /// %QChoice storing position and values for integer quantified views. Order of values
  /// is stored in a array such as it can be parse in a home made way.
  class GECODE_VTABLE_EXPORT QPosValuesOrderChoice : public PosValuesChoice {
  protected:
    /// The order of the alternatives to parse the domain in a customize way
    std::vector<unsigned int> parseOrder;
  public:
    /// Initialize choice for brancher \a b, position \a p and view \a x
    QPosValuesOrderChoice(const Brancher& b, const Pos& p, const std::vector<unsigned int>& parseOrder, IntView x);
    /// Initialize choice for brancher \a b from archive \a e
    QPosValuesOrderChoice(const Brancher& b, unsigned int alt, Pos p, Archive& e);
    /// Return value to branch with for alternative \a a
    int val(unsigned int a) const;
    /// Report size occupied
    virtual size_t size(void) const;
    /// Deallocate
    virtual ~QPosValuesOrderChoice(void);
    /// Archive into \a e
    virtual void archive(Archive& e) const;
  };

  forceinline int
  QPosValuesOrderChoice::val(unsigned int a) const {
    return PosValuesChoice::val(parseOrder[a]);
  }

  // ---------------------------------------
  // QViewValuesOrderBrancher
  // ---------------------------------------

  template<int n, bool min>
  forceinline
  QViewValuesOrderBrancher<n,min>::
  QViewValuesOrderBrancher(Home home, AsyncAlgo& _aAlgo, ViewArray<IntView>& x, int _offset,
                      ViewSel<IntView>* vs[n], BranchFilter bf, IntVarValPrint vvp0)
    : ViewValuesBrancher<n,min>(home,x,vs,bf,vvp0), offset(_offset), aAlgo(_aAlgo) {
    home.notice(*this,AP_DISPOSE);
//ICI    domainsParseOrder.init(x.size());
//ICI    receiver.setNbVars(offset + x.size());
//ICI    for (int i=0; i < x.size(); i++) {
//ICI      unsigned int k = 0;
//ICI      std::vector<TVal> domain;
//ICI      assert(x[i].range());
//ICI      for (int j=x[i].min(); j <= x[i].max(); j++, k++) {
//ICI        domainsParseOrder[i].push_back(ValOrder(j,k));
//ICI        domain.push_back(TVal(j));
//ICI      }
//ICI      SIBus::instance().sendDomainDescription(i+offset,domain);
//ICI    }
  }

  template<int n, bool min>
  BrancherHandle
  QViewValuesOrderBrancher<n,min>::post(Home home, AsyncAlgo& aAlgo, ViewArray<IntView>& x, int offset,
                                         ViewSel<IntView>* vs[n],
                                         BranchFilter bf, IntVarValPrint vvp) {
    return *new (home) QViewValuesOrderBrancher<n,min>(home,aAlgo,x,offset,vs,bf,vvp);
  }

  template<int n, bool min>
  forceinline
  QViewValuesOrderBrancher<n,min>::
  QViewValuesOrderBrancher(Space& home, bool shared, QViewValuesOrderBrancher& b)
     : ViewValuesBrancher<n,min>(home,shared,b), offset(b.offset), aAlgo(b.aAlgo) {
//ICI    domainsParseOrder.update(home,shared,b.domainsParseOrder);
  }

  template<int n, bool min>
  Actor*
  QViewValuesOrderBrancher<n,min>::copy(Space& home, bool shared) {
    return new (home) QViewValuesOrderBrancher<n,min>(home,shared,*this);
  }

//ICI  forceinline bool orderTest(Tuple i, Tuple j) { return (i.v2 < j.v2); }

  template<int n, bool min>
  const Choice*
  QViewValuesOrderBrancher<n,min>::choice(Space& home) {
    Pos p = this->pos(home);
//ICI    while (!receiver.emptySwapRequestFor(offset + p.pos)) {
//ICI      Tuple t = receiver.frontSwapRequestFor(offset + p.pos);
//ICI      unsigned int k = domainsParseOrder[p.pos][t.v1].o;
//ICI      domainsParseOrder[p.pos][t.v1].o = domainsParseOrder[p.pos][t.v2].o;
//ICI      domainsParseOrder[p.pos][t.v2].o = k;
//ICI      receiver.popSwapRequestFor(offset + p.pos);
//ICI      SIBus::instance().sendSwapDone(offset + p.pos,t.v1,t.v2);
//ICI    }
//ICI
//ICI    std::vector<Tuple> alternativeAndOrder;
//ICI    ViewValues<IntView> vv(ViewBrancher<IntView,n>::view(p));
//ICI    typename std::vector<ValOrder>::const_iterator it = domainsParseOrder[p.pos].begin();
//ICI    unsigned int i = 0;
//ICI    while (vv()) {
//ICI      assert(it != domainsParseOrder[p.pos].end());
//ICI      for ( ; (*it).v != vv.val() ; ++it );
//ICI      alternativeAndOrder.push_back(Tuple(i,(*it).o));
//ICI      ++i; ++vv;
//ICI    }
//ICI    std::sort(alternativeAndOrder.begin(), alternativeAndOrder.end(), orderTest);
    std::vector<unsigned int> parseOrder;
//ICI    typename std::vector<Tuple>::const_iterator it2    = alternativeAndOrder.begin();
//ICI    typename std::vector<Tuple>::const_iterator it2End = alternativeAndOrder.end();
//ICI    for ( ; it2 != it2End; ++it2 ) parseOrder.push_back((*it2).v1);

    return new QPosValuesOrderChoice(*this,p,parseOrder,
                                     ViewBrancher<IntView,n>::view(p));
  }

  template<int n, bool min>
  const Choice*
  QViewValuesOrderBrancher<n,min>::choice(const Space& home, Archive& e) {
    (void) home;
    int p;
    unsigned int a;
    e >> p >> a;
    return new QPosValuesOrderChoice(*this,a,p,e);
  }

  template<int n, bool min>
  ExecStatus
  QViewValuesOrderBrancher<n,min>::commit(Space& home, const Choice& c, unsigned int a) {
    const QPosValuesOrderChoice& pvc
      = static_cast<const QPosValuesOrderChoice&>(c);
    IntView x(ViewBrancher<IntView,n>::view(pvc.pos()));
    unsigned int b = min ? a : (pvc.alternatives() - 1 - a);
    return me_failed(x.eq(home,pvc.val(b))) ? ES_FAILED : ES_OK;
  }

  template<int n, bool min>
  forceinline size_t
  QViewValuesOrderBrancher<n,min>::dispose(Space& home) {
    home.ignore(*this,AP_DISPOSE);
//ICI    domainsParseOrder.~SharedArray();
    ViewValuesBrancher<n,min>::dispose(home);
    return sizeof(*this);
  }

  forceinline
  QPosValuesOrderChoice::QPosValuesOrderChoice(const Brancher& b, const Pos& p, const std::vector<unsigned int>& _parseOrder, IntView x)
    : PosValuesChoice(b,p,x), parseOrder(_parseOrder)
  {
//ICI    assert( this->alternatives() == parseOrder.size() );
  }

  forceinline
  QPosValuesOrderChoice::QPosValuesOrderChoice(const Brancher& b, unsigned int a, Pos p,
                                               Archive& e)
    : PosValuesChoice(b,a,p,e) {
//ICI    int dpoSize = 0;
//ICI    e >> dpoSize;
//ICI    parseOrder.resize(dpoSize);
//ICI    for (int i=0; i < dpoSize; i++) e >> parseOrder[i];
  }

  forceinline size_t
  QPosValuesOrderChoice::size(void) const {
    return PosValuesChoice::size() + sizeof(QPosValuesOrderChoice) - sizeof(PosValuesChoice);
  }

  forceinline
  QPosValuesOrderChoice::~QPosValuesOrderChoice(void) {
  }

  forceinline void
  QPosValuesOrderChoice::archive(Archive& e) const {
    PosValuesChoice::archive(e);
//ICI    int dpoSize = (int)parseOrder.size();
//ICI    e << dpoSize;
//ICI    for (int i=0; i < dpoSize; i++) e << parseOrder[i];
  }

}}}

// STATISTICS: int-branch
