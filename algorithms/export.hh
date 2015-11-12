#ifndef __EXPORT__H_
/*
 * Configure linking
 *
 */
#if !defined(ALGORITHM_STATIC_LIB) && \
    (defined(__CYGWIN__) || defined(__MINGW32__) || defined(_MSC_VER))

#ifdef BUILD_ALGORITHM_LIB
#define ALGORITHM_EXPORT __declspec( dllexport )
#else
#define ALGORITHM_EXPORT __declspec( dllimport )
#endif

#else

#ifdef ALGORITHM_GCC_HAS_CLASS_VISIBILITY
#define ALGORITHM_EXPORT __attribute__ ((visibility("default")))
#else
#define ALGORITHM_EXPORT
#endif

#endif

#endif
