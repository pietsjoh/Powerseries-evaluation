"""Will be deprecated.
"""
import os

def cd(dirPath):
    rval = os.system( "cd " + dirPath)

def rmdir(dirPath):
    rval = os.system("rm -rd " + dirPath)

def rm(dirPath):
    rval = os.system("rm " + dirPath)

def cp(filePath, targetPath):
    rval = os.system("cp " + filePath + " " + targetPath)

def mv(oldFilePath, newFilePath):
    rval = os.system("mv " + oldFilePath + " " + newFilePath)