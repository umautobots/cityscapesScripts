#!/usr/bin/python
#
# The evaluation script for pixel-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below 
# and set the required environment variables as needed, such that 
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# Note that the script is a lot faster, if you enable cython support.
# WARNING: Cython only tested for Ubuntu 64bit OS.
# To enable cython, run
# setup.py build_ext --inplace
#
# To run this script, make sure that your results are images,
# where pixels encode the class IDs as defined in labels.py.
# Note that the regular ID is used, not the train ID.
# Further note that many classes are ignored from evaluation.
# Thus, authors are not expected to predict these classes and all
# pixels with a ground truth label that is ignored are ignored in
# evaluation.

# python imports
from __future__ import print_function
import os, sys
import platform
import fnmatch

from multiprocessing import Pool, cpu_count
import signal


try:
    from itertools import izip
except ImportError:
    izip = zip

# Cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from csHelpers import *

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False


###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
#
# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable CITYSCAPES_RESULTS
#   - environment variable CITYSCAPES_DATASET/results
#   - ../../results/"
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
# <city>_123456_123456*.png
# for a ground truth filename
# <city>_123456_123456_gtFine_labelIds.png
def getPrediction( args, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not args.predictionPath:
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        elif 'CITYSCAPES_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['CITYSCAPES_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        args.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    csFile = getCsFileInfo(groundTruthFile)
    filePattern = "{}_{}_{}*.png".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()

# Where to look for Cityscapes
if 'CITYSCAPES_DATASET' in os.environ:
    args.cityscapesPath = os.environ['CITYSCAPES_DATASET']
else:
    args.cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

if 'CITYSCAPES_EXPORT_DIR' in os.environ:
    export_dir = os.environ['CITYSCAPES_EXPORT_DIR']
    if not os.path.isdir(export_dir):
        raise ValueError("CITYSCAPES_EXPORT_DIR {} is not a directory".format(export_dir))
    args.exportFile = "{}/resultPixelLevelSemanticLabeling.json".format(export_dir)
else:
    args.exportFile = os.path.join(args.cityscapesPath, "evaluationResults", "resultPixelLevelSemanticLabeling.json")
# Parameters that should be modified by user
args.groundTruthSearch  = os.path.join( args.cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_labelIds.png" )

# Remaining params
args.evalInstLevelScore = True
args.evalPixelAccuracy  = False
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ""
args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False

args.avgClassSize       = {
    "bicycle"    :  4672.3249222261 ,
    "caravan"    : 36771.8241758242 ,
    "motorcycle" :  6298.7200839748 ,
    "rider"      :  3930.4788056518 ,
    "bus"        : 35732.1511111111 ,
    "train"      : 67583.7075812274 ,
    "car"        : 12794.0202738185 ,
    "person"     :  3462.4756337644 ,
    "truck"      : 27855.1264367816 ,
    "trailer"    : 16926.9763313609 ,
}

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


#########################
# Methods
#########################


# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(args):
    args.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.id)
    maxId = max(args.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

def generateInstanceStats():
    instanceStats = {}
    instanceStats["classes"   ] = {}
    instanceStats["categories"] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats["classes"][label.name] = {}
            instanceStats["classes"][label.name]["tp"] = 0.0
            instanceStats["classes"][label.name]["tpWeighted"] = 0.0
            instanceStats["classes"][label.name]["fn"] = 0.0
            instanceStats["classes"][label.name]["fnWeighted"] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats["categories"][category] = {}
        instanceStats["categories"][category]["tp"] = 0.0
        instanceStats["categories"][category]["tpWeighted"] = 0.0
        instanceStats["categories"][category]["fn"] = 0.0
        instanceStats["categories"][category]["fnWeighted"] = 0.0
        instanceStats["categories"][category]["labelIds"] = labelIds

    return instanceStats


# Get absolute or normalized value from field in confusion matrix.
def getMatrixFieldValue(confMatrix, i, j, args):
    if args.normalized:
        rowSum = confMatrix[i].sum()
        if (rowSum == 0):
            return float('nan')
        return float(confMatrix[i][j]) / rowSum
    else:
        return confMatrix[i][j]

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    labelName = id2label[label].name
    if not labelName in instStats["classes"]:
        return float('nan')

    tp = instStats["classes"][labelName]["tpWeighted"]
    fn = instStats["classes"][labelName]["fnWeighted"]
    # false postives computed as above
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate prior for a particular class id.
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, args):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, args):
    # All labels in this category
    labels = category2labels[category]
    # The IDs of all valid labels in this category
    labelIds = [label.id for label in labels if not label.ignoreInEval and label.id in args.evalLabels]
    # If there are no valid labels, then return NaN
    if not labelIds:
        return float('nan')

    # the number of true positive pixels for this category
    # this is the sum of all entries in the confusion matrix
    # where row and column belong to a label ID of this category
    tp = np.longlong(confMatrix[labelIds,:][:,labelIds].sum())

    # the number of false negative pixels for this category
    # that is the sum of all rows of labels within this category
    # minus the number of true positive pixels
    fn = np.longlong(confMatrix[labelIds,:].sum()) - tp

    # the number of false positive pixels for this category
    # we count the column sum of all labels within this category
    # while skipping the rows of ignored labels and of labels within this category
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, args):
    if not category in instStats["categories"]:
        return float('nan')
    labelIds = instStats["categories"][category]["labelIds"]

    tp = instStats["categories"][category]["tpWeighted"]
    fn = instStats["categories"][category]["fnWeighted"]

    # the number of false positive pixels for this category
    # same as above
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom


# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, args ):
    # write JSON result file
    wholeData = {}
    wholeData["confMatrix"] = confMatrix.tolist()
    wholeData["priors"] = {}
    wholeData["labels"] = {}
    for label in args.evalLabels:
        wholeData["priors"][id2label[label].name] = getPrior(label, confMatrix)
        wholeData["labels"][id2label[label].name] = label
    wholeData["classScores"] = classScores
    wholeData["classInstScores"] = classInstScores
    wholeData["categoryScores"] = categoryScores
    wholeData["categoryInstScores"] = categoryInstScores
    wholeData["averageScoreClasses"] = getScoreAverage(classScores, args)
    wholeData["averageScoreInstClasses"] = getScoreAverage(classInstScores, args)
    wholeData["averageScoreCategories"] = getScoreAverage(categoryScores, args)
    wholeData["averageScoreInstCategories"] = getScoreAverage(categoryInstScores, args)

    if perImageStats:
        wholeData["perImageScores"] = perImageStats

    return wholeData

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportFile)

# Print confusion matrix
def printConfMatrix(confMatrix, args):
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "))

    # print label names
    print("\b{text:>{width}} |".format(width=13, text=""), end=' ')
    for label in args.evalLabels:
        print("\b{text:^{width}} |".format(width=args.printRow, text=id2label[label].name[0]), end=' ')
    print("\b{text:>{width}} |".format(width=6, text="Prior"))

    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "))

    # print matrix
    for x in range(0, confMatrix.shape[0]):
        if (not x in args.evalLabels):
            continue
        # get prior of this label
        prior = getPrior(x, confMatrix)
        # skip if label does not exist in ground truth
        if prior < 1e-9:
            continue

        # print name
        name = id2label[x].name
        if len(name) > 13:
            name = name[:13]
        print("\b{text:>{width}} |".format(width=13,text=name), end=' ')
        # print matrix content
        for y in range(0, len(confMatrix[x])):
            if (not y in args.evalLabels):
                continue
            matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, args)
            print(getColorEntry(matrixFieldValue, args) + "\b{text:>{width}.2f}  ".format(width=args.printRow, text=matrixFieldValue) + args.nocol, end=' ')
        # print prior
        print(getColorEntry(prior, args) + "\b{text:>{width}.4f} ".format(width=6, text=prior) + args.nocol)
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end=' ')

# Print intersection-over-union scores for all classes.
def printClassScores(scoreList, instScoreList, args):
    if (args.quiet):
        return
    print(args.bold + "classes          IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for label in args.evalLabels:
        if (id2label[label].ignoreInEval):
            continue
        labelName = str(id2label[label].name)
        iouStr = getColorEntry(scoreList[labelName], args) + "{val:>5.3f}".format(val=scoreList[labelName]) + args.nocol
        niouStr = getColorEntry(instScoreList[labelName], args) + "{val:>5.3f}".format(val=instScoreList[labelName]) + args.nocol
        print("{:<14}: ".format(labelName) + iouStr + "    " + niouStr)

# Print intersection-over-union scores for all categorys.
def printCategoryScores(scoreDict, instScoreDict, args):
    if (args.quiet):
        return
    print(args.bold + "categories       IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for categoryName in scoreDict:
        if all( label.ignoreInEval for label in category2labels[categoryName] ):
            continue
        iouStr  = getColorEntry(scoreDict[categoryName], args) + "{val:>5.3f}".format(val=scoreDict[categoryName]) + args.nocol
        niouStr = getColorEntry(instScoreDict[categoryName], args) + "{val:>5.3f}".format(val=instScoreDict[categoryName]) + args.nocol
        print("{:<14}: ".format(categoryName) + iouStr + "    " + niouStr)



# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args):
    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")

    if not args.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    confMatrix    = generateMatrix(args)
    instStats     = generateInstanceStats()
    perImageStats = {}
    nbPixels      = 0

    def worker_init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    parallelism = cpu_count()
    num_images = len(predictionImgList)
    chunksize = int(math.ceil(num_images / parallelism))
    pool = Pool(processes=parallelism, initializer=worker_init)
    tasks = [pool.apply_async(
        eval_image_list,
        args=(
            predictionImgList[i: min(num_images, i+chunksize)],
            groundTruthImgList[i: min(num_images, i+chunksize)],
            confMatrix.shape
        ),
        kwds={'worker_num': worker_num, 'worker_index': i}) for worker_num, i in enumerate(range(0, num_images, chunksize))]
    results_list = [task.get() for task in tasks]

    for results in results_list:
        nbPixels += results['nb_pixels']

        # accumulate conf_matrix
        confMatrix += results['conf_matrix']

        # accumulate instance stats
        for class_label, d in results['instance_stats']['classes'].items():
            root_d = instStats['classes'][class_label]
            root_d['tp'] += d['tp']
            root_d['tpWeighted'] += d['tpWeighted']
            root_d['fn'] += d['fn']
            root_d['fnWeighted'] += d['fnWeighted']
        for category, d in results['instance_stats']['categories'].items():
            root_d = instStats['categories'][category]
            root_d['tp'] += d['tp']
            root_d['tpWeighted'] += d['tpWeighted']
            root_d['fn'] += d['fn']
            root_d['fnWeighted'] += d['fnWeighted']

        # accumulate per image stats
        for pred_path, d in results['per_image_stats'].items():
            root_d = perImageStats[pred_path]
            root_d['nbNotIgnoredPixels'] = root_d.get('nbNotIgnoredPixels', 0) + d['nbNotIgnoredPixels']
            root_d['nbCorrectPixels'] = root_d.get('nbCorrectPixels', 0) + d['nbCorrectPixels']

    if not args.quiet:
        print("\n")

    # sanity check
    if confMatrix.sum() != nbPixels:
        printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

    # print confusion matrix
    if (not args.quiet):
        printConfMatrix(confMatrix, args)

    # Calculate IOU scores on class level from matrix
    classScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)

    # Calculate instance IOU scores on class level from matrix
    classInstScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        print("")
        printClassScores(classScoreList, classInstScoreList, args)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classInstScoreList , args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # Calculate IOU scores on category level from matrix
    categoryScoreList = {}
    for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,args)

    # Calculate instance IOU scores on category level from matrix
    categoryInstScoreList = {}
    for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        printCategoryScores(categoryScoreList, categoryInstScoreList, args)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryInstScoreList, args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # write result file
    allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, args )
    writeJSONFile( allResultsDict, args)

    # return confusion matrix
    return allResultsDict

def eval_image_list(pred_images, ground_truth_images, conf_matrix_shape, worker_num=0, worker_index=0):
    conf_matrix = np.zeros(conf_matrix_shape, dtype=np.ulonglong)
    instance_stats = generateInstanceStats()
    per_image_stats = {}
    nb_pixels = 0

    # Evaluate all pairs of images and save them into a matrix
    num_images = len(pred_images)
    for i in range(num_images):
        print("worker[{}] evaluating {} out of {} at index {}".format(worker_num, i, num_images, worker_index + i))
        sys.stdout.flush()
        predictionImgFileName = pred_images[i]
        groundTruthImgFileName = ground_truth_images[i]
        #print "Evaluate ", predictionImgFileName, "<>", groundTruthImgFileName
        results = evaluate_pair(predictionImgFileName, groundTruthImgFileName,
                               conf_matrix_shape = conf_matrix.shape,
                               evalLabels=args.evalLabels,
                               avgClassSize=args.avgClassSize,
                               evalInstLevelScore=args.evalInstLevelScore,
                               evalPixelAccuracy=args.evalPixelAccuracy)
        nb_pixels += results['nb_pixels']

        # accumulate conf_matrix
        conf_matrix += results['conf_matrix']

        # accumulate instance stats
        for class_label, d in results['instance_stats']['classes'].items():
            root_d = instance_stats['classes'][class_label]
            root_d['tp'] += d['tp']
            root_d['tpWeighted'] += d['tpWeighted']
            root_d['fn'] += d['fn']
            root_d['fnWeighted'] += d['fnWeighted']
        for category, d in results['instance_stats']['categories'].items():
            root_d = instance_stats['categories'][category]
            root_d['tp'] += d['tp']
            root_d['tpWeighted'] += d['tpWeighted']
            root_d['fn'] += d['fn']
            root_d['fnWeighted'] += d['fnWeighted']

        # accumulate per image stats
        for pred_path, d in results['per_image_stats'].items():
            root_d = per_image_stats[pred_path]
            root_d['nbNotIgnoredPixels'] = root_d.get('nbNotIgnoredPixels', 0) + d['nbNotIgnoredPixels']
            root_d['nbCorrectPixels'] = root_d.get('nbCorrectPixels', 0) + d['nbCorrectPixels']

    return {
        'nb_pixels': nb_pixels,
        'conf_matrix': conf_matrix,
        'instance_stats': instance_stats,
        'per_image_stats': per_image_stats,
    }


# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluate_pair(prediction_path, ground_truth_path,
                  conf_matrix_shape,
                  evalLabels, avgClassSize, evalInstLevelScore, evalPixelAccuracy):
    # Loading all resources for evaluation.
    try:
        prediction_image = Image.open(prediction_path)
        prediction_np = np.array(prediction_image)
    except:
        printError("Unable to load " + prediction_path)
    try:
        ground_truth_img = Image.open(ground_truth_path)
        ground_truth_np = np.array(ground_truth_img)
    except:
        printError("Unable to load " + ground_truth_path)
    # load ground truth instances, if needed
    if evalInstLevelScore:
        ground_truth_instance_img_file_name = ground_truth_path.replace("labelIds", "instanceIds")
        try:
            instance_img = Image.open(ground_truth_instance_img_file_name)
            instance_np = np.array(instance_img)
        except:
            printError("Unable to load " + ground_truth_instance_img_file_name)

    # Check for equal image sizes
    if prediction_image.size[0] != ground_truth_img.size[0]:
        printError("Image widths of " + prediction_path + " and " + ground_truth_path + " are not equal.")
    if prediction_image.size[1] != ground_truth_img.size[1]:
        printError("Image heights of " + prediction_path + " and " + ground_truth_path + " are not equal.")
    if len(prediction_np.shape) != 2:
        printError("Predicted image has multiple channels.")

    img_width = prediction_image.size[0]
    img_height = prediction_image.size[1]
    nb_pixels = img_width * img_height

    conf_matrix = np.zeros(conf_matrix_shape, dtype=np.ulonglong)
    instance_stats = generateInstanceStats()
    per_image_stats = {}

    # the slower python way
    for (ground_truth_img_pixel, prediction_img_pixel) in izip(ground_truth_img.getdata(), prediction_image.getdata()):
        if (not ground_truth_img_pixel in evalLabels):
            printError("Unknown label with id {:}".format(ground_truth_img_pixel))

        conf_matrix[ground_truth_img_pixel][prediction_img_pixel] += 1

    if evalInstLevelScore:
        # Generate category masks
        category_masks = {}
        for category in instance_stats["categories"]:
            category_masks[category] = np.in1d(
                prediction_np,
                instance_stats["categories"][category]["labelIds"]).reshape(
                prediction_np.shape)

        inst_list = np.unique(instance_np[instance_np > 1000])
        for instId in inst_list:
            label_id = int(instId / 1000)
            label = id2label[label_id]
            if label.ignoreInEval:
                continue

            mask = instance_np == instId
            inst_size = np.count_nonzero(mask)

            tp = np.count_nonzero(prediction_np[mask] == label_id)
            fn = inst_size - tp

            weight = avgClassSize[label.name] / float(inst_size)
            tp_weighted = float(tp) * weight
            fn_weighted = float(fn) * weight

            instance_stats["classes"][label.name]["tp"] += tp
            instance_stats["classes"][label.name]["fn"] += fn
            instance_stats["classes"][label.name]["tpWeighted"] += tp_weighted
            instance_stats["classes"][label.name]["fnWeighted"] += fn_weighted

            category = label.category
            if category in instance_stats["categories"]:
                cat_tp = np.count_nonzero(np.logical_and(mask, category_masks[category]))
                cat_fn = inst_size - cat_tp

                cat_tp_weighted = float(cat_tp) * weight
                cat_fn_weighted = float(cat_fn) * weight

                instance_stats["categories"][category]["tp"] += cat_tp
                instance_stats["categories"][category]["fn"] += cat_fn
                instance_stats["categories"][category]["tpWeighted"] += cat_tp_weighted
                instance_stats["categories"][category]["fnWeighted"] += cat_fn_weighted

    if evalPixelAccuracy:
        not_ignored_labels = [l for l in evalLabels if not id2label[l].ignoreInEval]
        not_ignored_pixels = np.in1d(ground_truth_np, not_ignored_labels, invert=True).reshape(ground_truth_np.shape)
        erroneous_pixels = np.logical_and(not_ignored_pixels, (prediction_np != ground_truth_np))
        per_image_stats[prediction_path] = {}
        per_image_stats[prediction_path]["nbNotIgnoredPixels"] = np.count_nonzero(not_ignored_pixels)
        per_image_stats[prediction_path]["nbCorrectPixels"] = np.count_nonzero(erroneous_pixels)

    return {
        'nb_pixels': nb_pixels,
        'conf_matrix': conf_matrix,
        'instance_stats': instance_stats,
        'per_image_stats': per_image_stats,
    }

# The main method
def main(argv):
    global args

    predictionImgList = []
    groundTruthImgList = []

    # the image lists can either be provided as arguments
    if (len(argv) > 3):
        for arg in argv:
            if ("gt" in arg or "groundtruth" in arg):
                groundTruthImgList.append(arg)
            elif ("pred" in arg):
                predictionImgList.append(arg)
    # however the no-argument way is prefered
    elif len(argv) == 0:
        # use the ground truth search string specified above
        groundTruthImgList = glob.glob(args.groundTruthSearch)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append( getPrediction(args,gt) )

    # evaluate
    evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
