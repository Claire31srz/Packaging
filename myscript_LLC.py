# Part 0: Libraries
#import 'requirements.txt'
import argparse
import logging
import os

from LL.LLC import *

# Part 1: Parser
parser = argparse.ArgumentParser()
parser.add_argument("--drop", type=float,
                    help="dropout rate")
parser.add_argument("--width", type=int,
                    help="First layer number of convolution filters")
parser.add_argument("--depth", type=int,
                    help="Number of convolution layer in the network")
parser.add_argument("--n_classes", type=int,
                    help="Number of classes in the dataset")
parser.add_argument("--n_epochs", type=int,
                    help="Number of epochs needed to train the model")
parser.add_argument("--batch_size", type=int,
                    help="Size of the batch needed to divid the epochs into it")
parser.add_argument("--output", type=str,
                    help="path to output file")

args = parser.parse_args()

# Part 2: Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("myscript.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

#Part 3: Errors
class Error(Exception):
    pass

class IncorrectOutputPath(Error):
    pass

class IncorrectCreatedParameters(Error):
    pass

class IncorrectModelParameters(Error):
    pass


# Part 4: Handling errors
    
def folder(inputfile, model):
    if os.path.exists(inputfile):
        pass
    else:
        raise IncorrectOutputPath ('Impossible to find the folder {}. Please enter a folder that exists.'.format(inputfile))
    
def parameter(dataset, n_epochs, batch_size, n_classes, drop, depth, width):
    if isinstance(n_epochs, int) & (n_epochs > 0):
        if isinstance(batch_size, int) & (batch_size > 0):
            if isinstance(n_classes, int) & (n_classes > 0):
                if isinstance(drop, float) & (drop > 0):
                    if isinstance(width, int) & (width > 0):
                        if isinstance(depth, int) & (depth > 0):
                            model_classifier = fit_model_on(dataset,n_epochs = n_epochs, batch_size = batch_size, n_classes = n_classes)
                        else: raise IncorrectCreatedParameters('The parameter depth = {} should be a positive integer. Please enter correct values for the depth.'.format(depth))
                    else: raise IncorrectCreatedParameters('The parameter width = {} should be a positive integer. Please enter correct values for the width.'.format(width))
                else: raise IncorrectCreatedParameters('The parameter depth = {} should be a positive integer. Please enter correct values for the depth.'.format(depth))
            else: raise IncorrectModelParameters('The parameter drop = {} should be positive float. Please enter a correct value.'.format(drop))
        else: raise IncorrectModelParameters('The parameter batch_size = {} should be positive integer. Please enter a correct value.'.format(batch_size))
    else: raise IncorrectModelParameters('The parameter n_epochs = {} should be positive integer. Please enter a correct value.'.format(n_epochs))
    return model_classifier

if __name__ == '__main__':
    logger.info("Execution starting. No error detected.")
    try:
        classifier = parameter(dataset = 'cifar10', n_epochs = args.n_epochs, 
                                batch_size = args.batch_size, 
                                n_classes = args.n_classes,
                                drop=args.drop,
                                depth = args.depth,
                                width = args.width)
        logger.info('The training is done. No error detected.')
    except (IncorrectModelParameters, IncorrectCreatedParameters) as error:
        logger.info('Error detected.')
        logger.warning(str(error))
    try:
        folder(model=model_classifier, intputfile = args.output)
        logger.info('Models and weights have been saved. No error detected.')
    except IncorrectOutputPath as error:
        logger.info('Error detected.')
        logger.warning(str(error))
    logger.info('Execution done. No error detected.')
                                                        

    