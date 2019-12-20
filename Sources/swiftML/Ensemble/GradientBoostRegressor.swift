//
//  GradientBoostRegressor.swift
//  DecisionTree
//
//  Created by Victor A on 8/9/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class GradientBoostRegressor {
    
    /// training data
    public var data: [[String]]
    /// column number of target var
    public var target: Int
    /// residuals
    public var residualData: [[String]]
    /// root for the regressor
    public var root: Float
    /// decision trees
    public var trees: [DecisionTree]
    /// iterations/num of trees to be created
    public var limit: Int
    /// learning rate for GBR
    public var learningRate: Float
    /// choice of regression vs. classification
    public var using: String
    
    ///initializer for AdaBoost Classifier
    /// - Parameters:
    ///    - data: data with labels
    ///    - target: column number of the labels
    ///    - Till: number of stumps to be generated
    ///    - learningRate -> a float for learning rate
    ///    - using: max infoGain or min gini impurity for tree making
    public init (data: [[String]], target: Int, till: Int, learningRate: Float = 0.1, using: String) {
        self.data = data
        self.target = target
        self.limit = till
        self.root = (Float)(0.0)
        self.learningRate = learningRate
        self.using = using
        
        for i in 1 ... data.count-1 {
            self.root += (Float)(data[i][target])!
        }
        
        self.root = self.root/(Float)(data.count-1)
        self.residualData = data
        
        for i in 1 ... data.count-1 {
            let residual = (Float)(data[i][target])!-self.root
            self.residualData[i][target] = (String)(residual)
        }
        
        self.trees = [DecisionTree]()
    }
    
    /// Creates the trees
    /// - Parameters: None
    /// - Returns: None
    public func boost(){
        for _ in 1 ... self.limit {
            let tree = DecisionTree(data: self.residualData, target: self.target, perform: "regression", using : self.using)
            self.trees.append(tree)
            self.updateResidualData()
        }
    }
    
    
    /// Updates residual data
    /// - Parameters: None
    /// - Returns: None
    public func updateResidualData() {
        let head = self.residualData[0]
        
        for i in 1 ... self.residualData.count-1 {
            var prediction = self.root
            
            for tree in trees {
                var example = [[String]]()
                example.append(head)
                example.append(self.residualData[i])
                prediction += self.learningRate*(Float)(tree.classify(example:example))!
            }
            let residual = (Float)(self.data[i][target])!-prediction
            self.residualData[i][target] = (String)(residual)
        }
        
    }
    
    /// Predicts value for an example by getting all tree decisions
    /// - Parameters:
    ///   -this: String array with feature header to be predicted
    /// - Returns: prediction as a string
    public func predict(this: [[String]]) -> String {
        var prediction = self.root
        for tree in trees {
            prediction += self.learningRate*(Float)(tree.classify(example:this))!
        }
        return (String)(prediction)
    }
    
    /// Scores the booster's accuracy on test data
    ///
    /// - Parameters:
    ///   - testData: test data as a 2D string array with feature header
    /// - Returns:
    ///   - accuracy of predictions as float
    ///   - predictions as string array
    public func score(testData: [[String]]) -> (Float, [String]){
        let head = testData[0]
        var residualsSquared: Float = 0
        var predictions = [String]()
        
        for i in 1 ... testData.endIndex-1 {
            var currentSample = [[String]]()
            currentSample.append(head)
            currentSample.append(testData[i])
            
            let prediction = predict(this: currentSample)
            predictions.append(prediction)
            let residual = (Float)(prediction)! - (Float)(testData[i][target])!
            
            residualsSquared = residualsSquared + (residual*residual)
        }
        
        let RMSE = pow(residualsSquared/(Float)(testData.endIndex-1), 0.5)
        
        return (RMSE, predictions)
        
    }
    
}
