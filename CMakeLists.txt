#
# Main authors:
#    Vincent Barichard <Vincent.Barichard@univ-angers.fr>
#
# Copyright:
#    Vincent Barichard, 2013
#
# Last modified:
#    $Date$ by $Author$
#    $Revision$
#
# This file is part of Quacode:
#    http://quacode.barichard.com
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(SIBus)

# determine, whether we want a static binary
SET(STATIC_LINKING FALSE CACHE BOOL "Build a static binary?")
# do we want static libraries?
IF(STATIC_LINKING)
    # When STATIC_LINKING is TRUE, than cmake looks for libraries ending
    # with .a. This is for linux only!
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    SET(CMAKE_EXE_LINKER_FLAGS "-static")
    # Remove flags to get rid off all the -Wl,Bydnamic
    SET(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)
    SET(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
    # Use static libs for Boost
    SET(Boost_USE_STATIC_LIBS ON)
    SET(Boost_USE_STATIC_RUNTIME ON)
ENDIF(STATIC_LINKING)

SET(CMAKE_VERBOSE_MAKEFILE TRUE)
SET(GECODE_SRC "${CMAKE_SOURCE_DIR}" CACHE PATH "Path where GeCode source is installed")
SET(GECODE_BIN "${GECODE_SRC}" CACHE PATH "Path where GeCode libs and binaries are installed")

SET(USE_SIBUS FALSE CACHE BOOL "Set to true to compil with SIBus")
SET(SIBUS_THREAD FALSE CACHE BOOL "Set to true to use thread in SIBus")
SET(LOG_AUDIT FALSE CACHE BOOL "Set to true to generate log output in parallel qcsp algorithm")

INCLUDE(CheckCXXCompilerFlag)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# ICI COMPILER QUACODE
# SI QUACODE S'EST COMPILE
IF(NOT GECODE_FOUND)
  MESSAGE(FATAL_ERROR "Gecode is needed, consider to install it")
ELSE (NOT GECODE_FOUND)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

  IF (CMAKE_COMPILER_IS_GNUCXX)
    ADD_DEFINITIONS(-Wall)
    ADD_DEFINITIONS(-Wextra)
    ADD_DEFINITIONS(-Wno-unused-local-typedefs)
    ADD_DEFINITIONS(-fimplement-inlines)
    ADD_DEFINITIONS(-fno-inline-functions)
    ADD_DEFINITIONS(-pipe)
    ADD_DEFINITIONS(-fPIC)
    SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
  ENDIF ()

  check_cxx_compiler_flag(-fvisibility=hidden HAVE_VISIBILITY_HIDDEN_FLAG)
  IF (HAVE_VISIBILITY_HIDDEN_FLAG)
    SET(GECODE_GCC_HAS_CLASS_VISIBILITY "/**/")
    ADD_DEFINITIONS(-fvisibility=hidden)
  ENDIF()

  IF (WIN32)
    SET(GECODE_THREADS_WINDOWS 1)
  ELSE ()
    SET(GECODE_THREADS_PTHREADS 1)
  ENDIF ()

  SET(SIBUS_HEADERS
    sibus/sibus.hh
    sibus/receivers/receiver-out.hh
    sibus/receivers/receiver-gecode.hh
    sibus/receivers/receiver-compil-certif.hh
    sibus/receivers/receiver-nodecount.hh
  )
  SET(SIBUS_HEADERS_NEED_THREADS
    sibus/receivers/receiver-network.hh
  )
  SET(SIBUS_HPP
    sibus/sibus.hpp
  )
  SET(SIBUS_SRCS
    sibus/sibus.cpp
    sibus/receivers/receiver-out.cpp
    sibus/receivers/receiver-network.cpp
    sibus/receivers/receiver-gecode.cpp
    sibus/receivers/receiver-compil-certif.cpp
    sibus/receivers/receiver-nodecount.cpp
  )
  SET(SIBUS_SRCS_NEED_THREADS
    sibus/receivers/receiver-network.cpp
  )
  SET(SIBUS_EXAMPLE_SRCS
    sibus/sample-from-file.cpp
  )
  SET(QCSP_LIB_HEADERS
    quacode/qcsp.hh
    quacode/qspaceinfo.hh
    quacode/support/dynamic-list.hh
    quacode/support/log.hh
    quacode/search/sequential/qpath.hh
    quacode/search/sequential/qdfs.hh
    quacode/search/parallel/qworkdata.hh
    quacode/search/parallel/qpath.hh
    quacode/search/parallel/qengine.hh
    quacode/search/parallel/qdfs.hh
    quacode/qint/qbool.hh
  )
  SET(QCSP_LIB_HPP
    quacode/qspaceinfo.hpp
    quacode/search/qdfs.hpp
    quacode/qint/watch.hpp
    quacode/qint/qbool/clause.hpp
    quacode/qint/qbool/eq.hpp
    quacode/qint/qbool/eqv.hpp
    quacode/qint/qbool/or.hpp
    quacode/qint/qbool/xor.hpp
    quacode/qint/qbool/xorv.hpp
  )
  SET(QCSP_LIB_SRCS
    quacode/qspaceinfo.cpp
    quacode/support/log.cpp
    quacode/search/qdfs.cpp
    quacode/search/sequential/qpath.cpp
    quacode/search/parallel/qpath.cpp
    quacode/search/parallel/qengine.cpp
    quacode/search/parallel/qdfs.cpp
    quacode/qint/qbool/eqv.cpp
    quacode/qint/qbool/qbool.cpp
    ${GECODE_SRC}/gecode/search/meta/nogoods.cpp
    ${GECODE_SRC}/gecode/search/parallel/engine.cpp
    ${GECODE_SRC}/gecode/search/parallel/path.cpp
  )
  SET(QCSP_EXAMPLES_SRCS
    quacode/examples/qbf.cpp
    quacode/examples/qdimacs.cpp
    quacode/examples/nim-fibo.cpp
    quacode/examples/matrix-game.cpp
    quacode/examples/connect-four.cpp
    quacode/examples/baker.cpp
    quacode/examples/speeddating.cpp
  )
  SET(FLATZINC_HEADERS
    quacode/flatzinc.hh
    quacode/flatzinc/ast.hh
    quacode/flatzinc/conexpr.hh
    quacode/flatzinc/flatzinc.hh
    quacode/flatzinc/option.hh
    quacode/flatzinc/plugin.hh
    quacode/flatzinc/registry.hh
    quacode/flatzinc/symboltable.hh
    quacode/flatzinc/varspec.hh
  )
  SET(FLATZINC_HPP
  )
  SET(FLATZINC_SRCS
    quacode/flatzinc/fzn-quacode.cpp
    quacode/flatzinc/flatzinc.cpp
    quacode/flatzinc/registry.cpp
  )

  IF (DISJUNCTION)
    SET(QCSP_EXAMPLES_SRCS ${QCSP_EXAMPLES_SRCS} quacode/examples/disjunction.cpp)
  ENDIF()

  SET(QCSP_EXAMPLES_SRCS_NEED_THREADS
    quacode/examples/baker-dbh.cpp
    quacode/examples/nim-fibo-dbh.cpp
    quacode/examples/matrix-game-dbh.cpp
  )
  SOURCE_GROUP("Hpp Files" REGULAR_EXPRESSION ".hpp")

  SET_SOURCE_FILES_PROPERTIES(${ALL_HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
  SET_SOURCE_FILES_PROPERTIES(${ALL_HPP} PROPERTIES HEADER_FILE_ONLY TRUE)

  SET(SIBUS_LIBS "")
  SET(EXTRA_SRCS "")
  IF (USE_SIBUS)
    ADD_DEFINITIONS(-DUSE_SIBUS)
    IF (SIBUS_THREAD)
      FIND_PACKAGE(Threads)
      IF(NOT Threads_FOUND)
        MESSAGE(FATAL_ERROR "Thread (pthread) library are needed, consider to install them or disable SIBus threads.")
      ELSE()
        FIND_PACKAGE(Boost 1.49.0 COMPONENTS system thread)
        IF(NOT Boost_FOUND)
          MESSAGE(STATUS "Try to find local Boost libraries.")
          SET(BOOST_LIBRARYDIR ${CMAKE_CURRENT_BINARY_DIR}/..)
          FIND_PACKAGE(Boost 1.49.0 COMPONENTS system thread)
        ENDIF()
        IF(NOT Boost_FOUND)
          MESSAGE(FATAL_ERROR "Boost thread and system library are needed, consider to install them or disable SIBus threads.")
        ELSE()
          INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
          ADD_DEFINITIONS(-DSIBUS_THREAD)
          ADD_LIBRARY(sibus ${SIBUS_SRCS} ${SIBUS_SRCS_NEED_THREADS} ${SIBUS_HEADERS} ${SIBUS_HEADERS_NEED_THREADS} ${SIBUS_HPP})
          SET(EXTRA_SRCS quacode/qint/branch/view-values.cpp)
          IF (BUILD_SHARED_LIBS)
            SET(SIBUS_LIBS -L${CMAKE_CURRENT_BINARY_DIR} -lsibus ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
          ELSE()
            SET(SIBUS_LIBS sibus ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
          ENDIF()
        ENDIF()
      ENDIF()
    ELSE()
      ADD_LIBRARY(sibus ${SIBUS_SRCS} ${SIBUS_HEADERS} ${SIBUS_HPP})
      IF (BUILD_SHARED_LIBS)
        SET(SIBUS_LIBS -L${CMAKE_CURRENT_BINARY_DIR} -lsibus)
      ELSE()
        SET(SIBUS_LIBS sibus)
      ENDIF()
    ENDIF ()
    SET_TARGET_PROPERTIES (sibus PROPERTIES DEFINE_SYMBOL  "BUILD_SIBUS")
    IF (HAVE_VISIBILITY_HIDDEN_FLAG)
      SET_TARGET_PROPERTIES(sibus PROPERTIES COMPILE_FLAGS "-DSIBUS_GCC_HAS_CLASS_VISIBILITY")
    ENDIF ()
  ENDIF ()

  IF (LOG_AUDIT)
    ADD_DEFINITIONS(-DLOG_AUDIT)
  ENDIF()

  ADD_LIBRARY(quacode ${QCSP_LIB_SRCS} ${EXTRA_SRCS} ${QCSP_LIB_HEADERS} ${QCSP_LIB_HPP})
  SET_TARGET_PROPERTIES(quacode PROPERTIES DEFINE_SYMBOL  "GECODE_BUILD_QCSP")
  SET(QCSP_LIB "")
  IF (BUILD_SHARED_LIBS)
    SET(QCSP_LIB -L${CMAKE_CURRENT_BINARY_DIR} -lquacode)
  ELSE()
    SET(QCSP_LIB quacode)
  ENDIF()

  # Add target for Quacode/FlatZinc
  FIND_PACKAGE(BISON 3.0.0)
  FIND_PACKAGE(FLEX)
  IF (BISON_FOUND AND FLEX_FOUND)
    BISON_TARGET(FLATZINCPARSER quacode/flatzinc/parser.yxx ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.cpp)
    FLEX_TARGET(FLATZINCSCANNER quacode/flatzinc/lexer.lxx  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
    ADD_FLEX_BISON_DEPENDENCY(FLATZINCSCANNER FLATZINCPARSER)
    ADD_EXECUTABLE(qfzn-quacode ${FLATZINC_SRCS} ${BISON_FLATZINCPARSER_OUTPUTS} ${FLEX_FLATZINCSCANNER_OUTPUTS})
    FIND_PACKAGE(Qt4 4.4 COMPONENTS QtCore)
    IF (QT_FOUND)
      GET_PROPERTY(DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
      SET(DIRS ${DIRS} ${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR})
      SET_TARGET_PROPERTIES(qfzn-quacode PROPERTIES INCLUDE_DIRECTORIES "${DIRS}")
      TARGET_LINK_LIBRARIES(qfzn-quacode ${QCSP_LIB} ${SIBUS_LIBS} ${GECODE_LIBRARIES} ${QT_QTCORE_LIBRARY} pthread)
    ELSE()
      TARGET_LINK_LIBRARIES(qfzn-quacode ${QCSP_LIB} ${SIBUS_LIBS} ${GECODE_LIBRARIES} pthread)
    ENDIF()

    # set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
    # to remove the additional -bdynamic from the linker line.
    IF(STATIC_LINKING)
      SET(CMAKE_EXE_LINKER_FLAGS "-static")
      SET_TARGET_PROPERTIES(qfzn-quacode PROPERTIES LINK_SEARCH_END_STATIC 1)
    ENDIF(STATIC_LINKING)

    IF (BUILD_SHARED_LIBS)
      ADD_DEPENDENCIES(qfzn-quacode quacode )
      IF (USE_SIBUS)
        ADD_DEPENDENCIES(qfzn-quacode sibus)
      ENDIF()
    ENDIF()
  ELSE()
    MESSAGE(STATUS "Can't find Flex and Bison, QFlatZinc will be not built.")
  ENDIF()


  # Add targets for examples
  FOREACH (example ${QCSP_EXAMPLES_SRCS})
    GET_FILENAME_COMPONENT(exampleBin ${example} NAME_WE)
    ADD_EXECUTABLE(${exampleBin} ${example})
    TARGET_LINK_LIBRARIES(${exampleBin} ${QCSP_LIB} ${SIBUS_LIBS} ${GECODE_LIBRARIES} pthread)

    # set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
    # to remove the additional -bdynamic from the linker line.
    IF(STATIC_LINKING)
        SET(CMAKE_EXE_LINKER_FLAGS "-static")
        SET_TARGET_PROPERTIES(${exampleBin} PROPERTIES LINK_SEARCH_END_STATIC 1)
    ENDIF(STATIC_LINKING)

    IF (BUILD_SHARED_LIBS)
      ADD_DEPENDENCIES(${exampleBin} quacode )
      IF (USE_SIBUS)
        ADD_DEPENDENCIES(${exampleBin} sibus)
      ENDIF()
    ENDIF()
  ENDFOREACH ()

  # Add targets for examples which need threads and SIBUS
  IF (USE_SIBUS)
    IF (SIBUS_THREAD)
      FOREACH (example ${QCSP_EXAMPLES_SRCS_NEED_THREADS})
        GET_FILENAME_COMPONENT(exampleBin ${example} NAME_WE)
        ADD_EXECUTABLE(${exampleBin} ${example})
        TARGET_LINK_LIBRARIES(${exampleBin} ${QCSP_LIB} ${SIBUS_LIBS} ${GECODE_LIBRARIES} pthread)

        # set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
        # to remove the additional -Bdynamic from the linker line.
        IF(STATIC_LINKING)
            SET(CMAKE_EXE_LINKER_FLAGS "-static")
            SET_TARGET_PROPERTIES(${exampleBin} PROPERTIES LINK_SEARCH_END_STATIC 1)
        ENDIF(STATIC_LINKING)

        IF (BUILD_SHARED_LIBS)
          ADD_DEPENDENCIES(${exampleBin} quacode)
          ADD_DEPENDENCIES(${exampleBin} sibus)
        ENDIF()
      ENDFOREACH ()
    ENDIF()
  ENDIF()

ENDIF(NOT GECODE_FOUND)

