# Example 3: Simple transfer entropy (TE) calculation on continuous-valued data using the (box) kernel-estimator TE calculator.

from jpype import *
import random, math

jarLocation = "lib/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation) 

# generate some random normalised data
numObservations = 1000
covariance = 0.4
sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]

# Destination array of random normals with partial correlation to previous value of sourceArray
destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]], \
                                             [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]


# Uncorrelated source array: 
sourceArray2 = [random.normalvariate(0,1) for r in range(numObservations)]

# Create a TE calculator and run it
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # normalise the individual variables
teCalc.initialise(1, 0.5) # use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))

# for copied source, should give something close to 1 bit
result = teCalc.computeAverageLocalOfObservations()

print("TE result %.4f bits; expected to be close to %.4f bits for these correlated Gaussians but biased upwards" % \
    (result, math.log(1/(1-math.pow(covariance,2)))/math.log(2)))

teCalc.initialise() # Initialise leaving the parameters the same
teCalc.setObservations(JArray(JDouble, 1)(sourceArray2), JArray(JDouble, 1)(destArray))
# For random source, it should give something close to 0 bits
result2 = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f bits; expected to be close to 0 bits for uncorrelated Gaussians but will be biased upwards" % \
    result2)