from jpype import *
import numpy
import sys

# Our python data file readers are a bit of a hack, python users will do better on this:
# sys.path.append("lib\jidt\demos\python")
import readFloatsFile

if (not isJVMStarted()):
    # Add JIDT jar library to the path
    jarLocation = "lib/jidt/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

# 0. Load/prepare the data:
dataRaw = readFloatsFile.readFloatsFile("D:\\codigos\\lab\\projetos PYTHON\\entropy_transfer_py\\data\\simulateEEG.csv")
# As numpy array:
data = numpy.array(dataRaw)
source = JArray(JDouble, 1)(data[:,1].tolist())
destination = JArray(JDouble, 1)(data[:,2].tolist())

# 1. Construct the calculator:
calcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
calc = calcClass()
# 2. Set any properties to non-default values:
# No properties were set to non-default values
# 3. Initialise the calculator for (re-)use:
calc.initialise()
# 4. Supply the sample data:
calc.setObservations(source, destination)
# 5. Compute the estimate:
result = calc.computeAverageLocalOfObservations()

print("TE_Kernel(col_1 -> col_2) = %.4f bits" %\
    (result))