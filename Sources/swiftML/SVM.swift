//
//  SVM.swift
//  SVM
//
//  Created by Victor A on 7/30/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class SVM {
    
    //arguments for the initializer or predict function
    public var arguments: String
    //the optimization problem
    public var problem: svm_problem
    //the parameters for the svm train
    public var parameter: svm_parameter
    //the fitted model returned by trainOn
    public var fittedModel: UnsafeMutablePointer<svm_model>
    //file path where the model File is stored
    public var modelFilePath: String
    
    
    /// Stores a SVM fitted on data
    ///
    /// - Parameters:
    ///   - settings: for svm train as per libSVM format
    ///   - dataFilePath: path to the data file to be trained on
    ///   - saveModelAt: path to where to save model at
    /// - Returns: nothing
    public init (settings: String = "", dataFilePath: String, saveModelAt: String){
        let function = "svm-train"
        self.arguments = function+" "+settings+" "+dataFilePath
        let args = self.arguments.components(separatedBy: " ")
        var argv = args.map { strdup($0) }
        let argc : Int32 = (Int32)(argv.count)
    
        trainOn(argc, &argv, UnsafeMutablePointer<Int8>(mutating: (saveModelAt as NSString).utf8String))
        
        self.modelFilePath = saveModelAt
        self.problem = prob
        self.parameter = param
        self.fittedModel = model
    }
    
    /// Returns SVM predictions for testing data
    ///
    /// - Parameters:
    ///   - testingData: path to the data file (libSVM format) to be tested on
    ///   - settings: for svm predict as per libSVM format
    ///   - saveAt: path to where to save predictions at
    /// - Returns: float array of predictions
    public func predict(testingData: String, settings: String = "", saveAt: String) -> [Float]{
        let function = "svm-predict"
        self.arguments = function+" "+settings+" "+testingData+" "+modelFilePath+" "+saveAt
        let args = self.arguments.components(separatedBy: " ")
        var argv = args.map { strdup($0) }
        let argc : Int32 = (Int32)(argv.count)
        predictWith(argc, &argv, fittedModel)
        let predictions = fileToTensor(path: saveAt)
        return predictions
    }
    
    /// Returns Float Array of data from a libSVM file
    ///
    /// - Parameters:
    ///   - path: to the file to be converted to Tensor
    /// - Returns: float array of data
    public func fileToTensor(path: String) -> [Float]{
        var predictions = [Float]()
        do {
            // Get the contents
            let contents = try NSString(contentsOfFile: path, encoding: String.Encoding.utf8.rawValue) as String
            let contentComponents = contents.components(separatedBy: "\n")
        
            for value in contentComponents {
                if((Float)(value) != nil){
                    predictions.append((Float)(value)!)
                }
            }
        }
        catch let error as NSError {
            print("Ooops! Something went wrong: \(error)")
        }
        
        return predictions
    }
    
}


/// Returns a libSVM file with data from a data & label tensor
///
/// - Parameters:
///   - path: to the where file to be saved
///   - data: 2D array of float data
///   - label: float array of label
/// - Returns: nothing
public func tensorToFile(path: String, data: [[Float]], label: [Float]){
    
    var contents = ""
    
    for i in 0..<data.count{
        contents = contents + (String)(label[i])
        for j in 0..<data[0].count{
            contents = contents + " " + (String)(j) + ":" + (String)(data[i][j])
        }
        contents = contents + "\n"
    }
    
    do {
        // Write contents to file
        try contents.write(toFile: path, atomically: false, encoding: String.Encoding.utf8)
    }
    catch let error as NSError {
        print("Ooops! Something went wrong: \(error)")
    }
    
    
}
