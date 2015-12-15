#ifndef __ASYNCALGOTYPES_H_
#define __ASYNCALGOTYPES_H_

#define TYPE_BOOL  0
#define TYPE_INT   1
// Information on type of a variable
typedef unsigned int TVarType;

#define CMP_NQ  0
#define CMP_EQ  1
#define CMP_LQ  2
#define CMP_LE  3
#define CMP_GQ  4
#define CMP_GR  5
// Information on comparison operator
typedef unsigned int TComparisonType;

namespace Gecode {
  #define EXISTS 0
  #define FORALL 1
  typedef unsigned int TQuantifier;
}

#endif
