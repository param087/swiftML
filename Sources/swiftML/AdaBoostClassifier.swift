//
//  AdaBoostClassifier.swift
//  Boosting
//
//  Created by Victor A on 8/7/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class AdaBoostClassifier {
    
    /// decision stumps for the booster
    public var stumps: [DecisionTree]
    /// weights computed during boosting
    public var weights: [Float]
    /// alphas stored during boosting
    public var alphas: [Float]
    /// errors used during boosting
    public var errors: [Float]
    /// number of stumps to be created
    public var iterations: Int
    /// training data
    public var data: [[String]]
    /// K value for boosting
    public var K: Float
    /// column number of target var
    public var target: Int
    /// choice of regression vs. classification
    public var using: String
    
    
    ///initializer for AdaBoost Classifier
    /// - Parameters:
    ///    - data: data with labels
    ///    - target: column number of the labels
    ///    - till: number of stumps to be generated
    ///    - using: info gain or gini Impurity
    public init (data: [[String]], target: Int, till: Int, using: String){
        self.data = data
        self.target = target
        let initialWeight = Float(1)/(Float)(data.count-1)
        self.weights = [Float](repeating: initialWeight, count: data.count-1)
        self.iterations = till
        self.stumps = [DecisionTree]()
        self.alphas = [Float]()
        self.errors = [Float]()
        let feature = Feature(data: data, column: target)
        self.K = (Float)(feature.values.count)
        self.using = using
    }
    
    /// Creates the stumps and fills weights
    /// - Parameters: None
    /// - Returns: None
    public func boost(){
        var stumpData = self.data
        for _ in 1...self.iterations {
            let stump = DecisionTree(data: stumpData, target: self.target, maxDepth: 1, perform: "classification", using : self.using)
            self.stumps.append(stump)
            computeError(stump: stump)
            stumpData = getWeightSampledData(from : stumpData)
        }
    }
    
    /// Computes Error for a given stump
    ///  - Parameters:
    ///    - stump: Decision Tree for which error needs to be computed
    ///  - Returns: Nothing
    public func computeError(stump: DecisionTree) {
        let examples = deleteColumn(data: self.data, column: self.target)
        let head = self.data[0]
        var error: Float = 0.0
        //let denominator : Float = self.weights.reduce(0, +)
        var correctIndices = [Int]()
        var incorrectIndices = [Int]()
        
        for i in 1 ... examples.count-1 {
            var example = [[String]]()
            example.append(head)
            example.append(examples[i])
            
            let stumpClassification = stump.classify(example:example)
            
            if stumpClassification == self.data[i][self.target] {
                correctIndices.append(i)
            } else {
                error += self.weights[i-1]
                incorrectIndices.append(i)
            }
            
        }
        
        let alpha = (log10((1-error)/error)+log10(self.K - 1))
        self.alphas.append(alpha)
        for i in correctIndices {
            self.weights[i-1] = self.weights[i-1]/(self.K*(1-error))
        }
        for j in incorrectIndices {
            self.weights[j-1] = self.weights[j-1]/(self.K*error)
        }
    }
    
    /// Returns new dataset with wieghted sampling based on previous errors
    /// - Parameters:
    ///   - from: data to sample from
    /// - Returns: sampled data
    public func getWeightSampledData(from: [[String]]) -> [[String]]{
        var sampledData = [[String]]()
        let head = self.data[0]
        sampledData.append(head)
        
        for _ in 1...data.count-1 {
            let index = Float.random(in: 0 ..< 1)
            var chosenExample : Int = 1
            var currentWeight : Float = self.weights[0]
            
            while currentWeight < index {
                if(chosenExample > data.count-1){
                    break;
                }
                currentWeight += self.weights[chosenExample]
                chosenExample += 1
            }
            
            sampledData.append(from[chosenExample])
        }
        
        return sampledData
        
    }
    
    /// Classfies an example by getting all stump classification and weighing them
    /// - Parameters:
    ///   -this: String array with feature header to be classified
    /// - Returns: classification as a stringß
    public func classify(this: [[String]]) -> String{
        let result = Feature(data: self.data, column : self.target)
        var resultProbabilities = [Float]()
        var resultValues = [String]()
        
        for value in result.values {
            var probability : Float = 0.0
            resultValues.append(value.name)
            
            for i in 0...self.stumps.count-1 {
                let stump = self.stumps[i]
                let stumpClassification = stump.classify(example:this)
                
                if stumpClassification == value.name {
                    probability += self.alphas[i]
                }
            }
            resultProbabilities.append(probability)
        }
    
        let highestProbabilityIndex = resultProbabilities.firstIndex(of: resultProbabilities.max()!)!
        let classification = resultValues[highestProbabilityIndex]
        return classification
    }
    
    
    /// Scores the booster's accuracy on test data
    ///
    /// - Parameters:
    ///   - testData: test data as a 2D string array with feature header
    /// - Returns:
    ///   - accuracy of classifications as float
    ///   - classifications as string array
    public func score(testData: [[String]]) -> (Float, [String]){
        let head = testData[0]
        var correctClassification = 0
        var classifications = [String]()
        
        for i in 1 ... testData.endIndex-1 {
            var currentSample = [[String]]()
            currentSample.append(head)
            currentSample.append(testData[i])
            
            let classification = classify(this: currentSample)
            classifications.append(classification)
            
            if classification == testData[i][target] {
                correctClassification = correctClassification + 1
            }
        }
        return ((Float)(correctClassification)/(Float)(testData.endIndex-1), classifications)
    }
    
}
