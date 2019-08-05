//
//  main.swift
//  SVM
//
//  Created by Victor A on 7/27/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation
import Darwin


let svm = SVM(settings : "-s 0 -c 5 -t 2 -g 0.5 -e 0.1",
              fileName: "/Users/victora/Downloads/libsvmTest/heart_scale",
              saveAt: "/Users/victora/Downloads/libsvmTest/")



/*

var options = "-s 0 -c 5 -t 2 -g 0.5 -e 0.1"
var fileName = "/Users/victora/Downloads/libsvmTest/heart_scale"
var function = "svm-train"

print(function+" "+options+" "+fileName)

 



print("Hello, World!")

//var arguments = "svm-scale -s /Users/victora/Desktop/holder/heart-scaled"

var arguments = "svm-scale -l -1 -u 1 -s /Users/victora/Desktop/holder/heart-scaled /Users/victora/Desktop/holder/heart-heart"

var args = arguments.components(separatedBy: " ")

var argv = args.map { strdup($0) }

print(argv)

var argc : Int32 = (Int32)(argv.count)

let outcome = scaleWith(argc, &argv)



var arguments = "svm-predict /Users/victora/Desktop/holder/fulltest/heart_test /Users/victora/Desktop/holder/fulltest/heart-heart.model /Users/victora/Desktop/holder/fulltest/heart-output"

var args = arguments.components(separatedBy: " ")

var argv = args.map { strdup($0) }

print(argv)

var argc : Int32 = (Int32)(argv.count)

let outcome = predictWith(argc, &argv)

 */

/*
var arguments = "svm-scale -l 0 -u 1 -s /Users/victora/Desktop/holder/heart-scaled /Users/victora/Desktop/holder/heart-heart > /Users/victora/Desktop/holder/heart-heart-scaled"

var args = arguments.components(separatedBy: " ")

var argv = args.map { strdup($0) }

print(argv)

var argc : Int32 = (Int32)(argv.count)

var outcome = scaleWith(argc, &argv)


arguments = "svm-scale -l 0 -u 1 -r /Users/victora/Desktop/holder/heart-scaled /Users/victora/Desktop/holder/heart-heart"

args = arguments.components(separatedBy: " ")

argv = args.map { strdup($0) }

print(argv)

argc = (Int32)(argv.count)

outcome = scaleWith(argc, &argv)


arguments = "svm-train -s 0 -c 5 -t 2 -g 0.5 -e 0.1 /Users/victora/Desktop/holder/heart-scaled"

args = arguments.components(separatedBy: " ")

print(args)

argv = args.map { strdup($0) }

print(argv)

argc = (Int32)(argv.count)

let path = "/Users/victora/Desktop/holder/"

outcome = trainOn(argc, &argv, UnsafeMutablePointer<Int8>(mutating: (path as NSString).utf8String))

 */
