Quacode is a quantified constraint satisfaction
problems (QCSP) solver based on Gecode.

Quacode have been developped by Vincent Barichard.
More info is available on http://quacode.barichard.com

This package provides "Quacode Driven By Heuristic" an
extension of Quacode for programmers who want to test
hybridization of a complete search algorithm for QCSP
with metaheuristics.

Quacode driven by heuristic, named QuacodeDBH for sake
of simplicity, is an extension of Quacode which allows
the search to be driven by another method run in parallel
in another thread. This second method, that we will call
heuristic, may be any algorithm designed by a programmer.
For example this package provides three algorithms:
- a "logger" which prints the data gather during the search
- a "dumb" algorithm which randomly changes the search direction
- a "MonteCarlo" type algorithm which drives the search according
to its results

QuacodeDBH is built for programmers who want to test their
metaheuristics. It is designed to easily allow programmers
to add their own algorithm.

I - Retrieve and build Gecode
-----------------------------
QuacodeDBH, as Quacode is based on Gecode. Quacode is provided
with the sources of Gecode but not QuacodeDBH. We must first
retrieve and build Gecode.
As the development of Gecode keeps moving forward you will find
on our web site a snapshot of the trunk of Gecode which works well
with QuacodeDBH. This package will be regularly updated according
to new incompatible updates of the Gecode trunk.

After downloading the Gecode sources and unpacking them (we name
path_to_Gecode_src the source directory of Gecode), go inside
the directory and follow the instructions to build Gecode. If you
don't need a custom installation of Gecode, a simple
'./configure && make' must work (on Linux based systems).
You do not have to run 'make install', you just have to remember
the path used and probably add to your environment variables:
    export LD_LIBRARY_PATH=".:path_to_Gecode_bin:$LD_LIBRARY_PATH"

Here path_to_Gecode_bin is the same directory as path_to_Gecode_src,
but advanced users may use different directories.

You can check that your Gecode installation works as expected by
launching an example provided with Gecode. For example you
can run './examples/queens'

Here, Gecode was built in Release mode, you can build a Debug
mode more useful for testing and debugging new stuff by
invoking './configure --enable-debug --enable-audit' instead
of the simple './configure'. Note that the programs will be
slower but can be debugged.


II - Retrieve and build QuacodeDBH
---------------------------------

To compile QuacodeDBH, you have to install cmake. To setup the
compilation process for your environment, you can launch
cmake by invoking
  cmake -DGECODE_SRC=path_to_Gecode_src -DGECODE_BIN=path_to_Gecode_bin .
in the toplevel QuacodeDBH source directory.

If you want to separate the sources from the binaries (which
is often a good choice), you can create another directory
for QuacodeDBH binaries. Then go inside and invoke:
  cmake -DGECODE_SRC=path_to_Gecode_src -DGECODE_BIN=path_to_Gecode_bin path_to_QuacodeDBH_src

Then you can compile the code by invoking
  make
in the toplevel QuacodeDBH build directory.

You can check that QuacodeDBH works as expected by going
to the build directory and launch a provided example.
You can run './examples/baker' and see what happen.

The following part is not mandatory, it is even not advised
if you want to programm and test new methods. It is only
needed when you want to deploy QuacodeDBH and your algorithms
on other computers. So if you don't need it, jump directly to
the next section.

By default, 'make install' will install all the files in
'/usr/local/bin', '/usr/local/lib' etc.  You can specify
an installation prefix other than '/usr/local' setting the
'CMAKE_INSTALL_PREFIX' option,
for instance 'cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME .'

After a successful compilation, you can install Quacode
library and examples by invoking
  make install
in the build directory.


III - Add its own algorithm to drive Quacode
--------------------------------------------

Adding its own algorithm is very easy. In the sources directory
of QuacodeDBH you will find a directory named 'algorithms'.
Go inside and copy file 'dumb.hh' to 'myAlgo.hh' and 'dumb.cpp'
to 'myAlgo.cpp'. Then open 'dumb.hh' and replace all occurences
of 'DumbAlgorithm' by 'MyAlgorithm' and do the same in 'dumb.cpp'.

Now you need to change a provided example to do use your own
algorithm. Open one of the example, for example 'examples/baker.cpp'.
At the top of the file, add a new 'include' line:
    #include <algorithms/myAlgo.hh>
Then go to the end of the file in the 'int main()' function.
In this function, you will find a line which instantiates the
heuristic algorithm object. For example a line like
'DumbAlgorithm aAlgo;'. Replace 'DumbAlgorithm' by 'MyAlgorithm'
and save the file.

Now go to the build directory of QuacodeDBH and re-run 'make'.
It will automatically add your algorithm MyAlgorithm to all examples
and the 'baker' example will now using your algorithm 'MyAlgorithm'.
You can check it by running again './examples/baker'.


IV - What can we do and how it works?
-------------------------------------

To add your algorithm you only have to copy some files in the
algorithms directory (see previous section).

To be used by Quacode, your algorithm has to inherit from the
class AsyncAlgo and must redefine some abstract functions.
There is four kinds of functions.

A - The functions that are automatically called by Quacode in
    the modeling stage. The search is not run yet and there is no
    other thread as the main one.
B - The functions that are automatically called by Quacode during
    the search. These functions are called in the Quacode thread
    so the source code of these functions must as fast as possible.
C - The functions that can be called by any algorithm in order to
    access or change mutual data during the search. These functions
    are thread safe and can be invoked at any time by your heuristic.
D - The function which is run in another thread when the search starts.
    This is inside this function that the programmer can add its own
    method.

See 'quacode/quacode/asyncalgo.hh' for more details about each function.

** Group A

These functions are called during the modeling stage in the Quacode thread.
Each function corresponds to a specific task:

newVarCreated is called each time a new variable is created in the problem.
  this function is dedicated to decision variables that are part of the
  binder.
newAuxVarCreated is called each time an auxiliary new variable is created
  in the problem. this function is dedicated to variables that are needed
  because of the problem decomposition.

Then each time a new constraint is added to the problem, a specific function
is called by Quacode. postedEq, postedAnd, postedOr, postedImp, postedXOr,
postedPlus, postedTimes and postedLinear are the name of the main useful
functions.

These functions are useful to gather data during the modeling stage before
the start of the search. It's up to the programmer to decide what is useful
or not.
Among these functions, only the two first ones (newVarCreated and newAuxVarCreated)
must be redefined as they are pure abstract functions. The others are
not mandatory but if they are used in the modeling stage and not redefined
in your algorithm, the program will stop with an error message.


** Group B

These functions are automatically called by Quacode during the
search to inform the heuristic of the advance of the search.
These methods are called in the Quacode thread and their
source code must be as fast as possible. It may be used to
store these data in your own data structure in order to use
them later by your algorithm. Each event occured during the
search will call a dedicated function:

newChoice is called each time the search develops a new node of the
  search tree
newPromisingScenario is called each time the search encounters a
  new promising scenario. It will backtrack if there is not yet
  a full winning strategy
strategyFound is called once, when the search ends with a winning
  strategy
newFailure is called each time the search finds a failure. It will
  backtrack if there is other alternative to explore
globalFailure is called once, when the search ends and no winning
  strategy has been found

All these functions are pure abstract functions and must be redefined
in your class even if they do nothing.

Be careful! These functions are not thread safe. They can not securely
access your own data structure if they are used in another thread.
For example, if these data structures are also used in your parallelTask
function (see Group D).


** Group C

These functions are thread safe, they can be called at anytime.
Let's focus on the 'swap' function. It is the only function
which will change the search direction of Quacode. Indeed, by
calling 'swap' on the domain of a variable you will change the
order in which the values are searched. 'swap' doesn't loose
the completness of the search as all values will be explored,
but only in a different order.

swap will swap two values of the domain of a variable so the search
  will not look over the values of this domain in the same order as
  before
getValue returns the nth value of the domain of a variable
copyDomain duplicate the domain of a variable in a vector

'swap' is the only function up to now which change the search
direction. Others will be added later in the same way.

** Group D

'parallelTask' is the main function run by Quacode in another
thread. It is started when the search starts after the modeling
stage. All the source code put in this function will be called
in the second thread in parallel to the search done by Quacode.
It is a pure abstract function and must be redefined in your class.

This is the place where to put your own source code. The thread
will be automatically destroyed by Quacode at the end of its search.
You don't need to manage the closing of the thread, but it can
be needed if you have to free some data structures you allocated
before. An example of how to do that is shown in the files
'montecarlo.hh' and 'montecarlo.cpp'.
The main idea is to add a mutex in the destructor which forces
Quacode to wait for the clean end of the 'parallelTask' function.
In the file 'montecarlo.cpp', this mutex is named 'mDestructor'
and it is acquired before the MonteCarlo starts and in the
desctructor. So, when Quacode will call the destructor, it will
forced to wait for the end of the MonteCarlo algorithm when
the mutex is released.

