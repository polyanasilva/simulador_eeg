from jpype import *
import random
import math

# Change location of jar to match yours:
jarLocation = "lib/jidt/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation) 

# Generate some random normalised data.
numObservations = 1000
covariance=0.4
# Source array of random normals:
sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]
# Destination array of random normals with partial correlation to previous value of sourceArray
destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]], \
                                             [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
# Uncorrelated source array:
sourceArray2 = [random.normalvariate(0,1) for r in range(numObservations)]
# Create a TE calculator and run it:
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.initialise(1) # Use history length 1 (Schreiber k=1)
teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
# Perform calculation with correlated source:
teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))
result = teCalc.computeAverageLocalOfObservations()
# Note that the calculation is a random variable (because the generated
#  data is a set of random variables) - the result will be of the order
#  of what we expect, but not exactly equal to it; in fact, there will
#  be a large variance around it.
print("TE result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" % \
    (result, math.log(1/(1-math.pow(covariance,2)))))
# Perform calculation with uncorrelated source:
teCalc.initialise() # Initialise leaving the parameters the same
teCalc.setObservations(JArray(JDouble, 1)(sourceArray2), JArray(JDouble, 1)(destArray))
result2 = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f nats; expected to be close to 0 nats for these uncorrelated Gaussians" % result2)

# We can also compute the local TE values for the time-series samples here:
#  (See more about utility of local TE in the CA demos)
localTE = teCalc.computeLocalOfPreviousObservations();
# Now: localTE is of type JArray, e.g. print(type(localTE)) will return <class 'jpype._jarray.double[]'>
#  You can access individual entries, e.g. localTE[1], and use functions such as sum():
print("Notice that the mean of locals, %.4f nats, equals the previous result" % \
    (sum(localTE)/(numObservations-1)));
# You can convert back to a native python list as follows if you wish:
localTEPython = [x for x in localTE];