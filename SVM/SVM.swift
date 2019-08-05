//
//  SVM.swift
//  SVM
//
//  Created by Victor A on 7/30/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class SVM {
    
    public var arguments : String
    public var problem : svm_problem
    public var parameter : svm_parameter
    public var fittedModel : UnsafeMutablePointer<svm_model>
    public var modelFilePath : String
    
    public init (settings : String, fileName: String, saveAt: String){
        let function = "svm-train"
        self.arguments = function+" "+settings+" "+fileName
        let args = self.arguments.components(separatedBy: " ")
        var argv = args.map { strdup($0) }
        let argc : Int32 = (Int32)(argv.count)
        self.modelFilePath = saveAt
        
        trainOn(argc, &argv, UnsafeMutablePointer<Int8>(mutating: (saveAt as NSString).utf8String))
        
        self.problem = prob
        self.parameter = param
        self.fittedModel = model
    }
    
    public func predict(testingData: String, settings: String, saveAt: String){
        let function = "svm-predict"
        self.arguments = function+" "+settings+" "+testingData+" "+modelFilePath+" "+saveAt
        
        let args = self.arguments.components(separatedBy: " ")
        var argv = args.map { strdup($0) }
        let argc : Int32 = (Int32)(argv.count)
        
        predictWith(argc, &argv)
    }
    
    
}

