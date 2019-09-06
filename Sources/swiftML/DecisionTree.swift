//
//  DecisionTree.swift
//  SwiftML
//
//  Created by Victor A on 8/15/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class DecisionTree {
    
    /// training data set whole as inputed by user in init
    public var originalDataSet: DataSet
    /// root node of the decision tree
    public var root: Node?
    /// max depth tree is grown
    public var maxDepth: Int
    /// regression or classification task
    public var perform: String
    /// column number of target var
    public var target: Int
    /// tolerance for regression
    public var tolerance: Float
    
    /// Creates original DataSet to be stored and grows decision tree
    ///  - Parameters:
    ///    - data: data with labels
    ///    - target: column number of label
    ///    - maxDepth: max depth tree is grown
    ///    - perform: regression or classification
    ///    - using: infoGain or giniIndex
    ///    - tolerance: for regression only
    ///  - Returns: DecisionTree
    public init (data: [[String]],
                 target: Int,
                 maxDepth: Int = 9999,
                 perform: String,
                 using: String,
                 tolerance: Float = 0.1) {
        self.originalDataSet = DataSet(data: data,
                                       target: target)
        self.maxDepth = maxDepth
        self.perform = perform
        self.target = target
        self.tolerance = tolerance
        if perform == "regression" {
            if using == "infogain" {
                self.root = id3R(dataset: self.originalDataSet, depth: 0)
            } else if using == "gini" {
                self.root = giniR(dataset: self.originalDataSet, depth: 0)
            } else {
                print("invalid operation requested")
            }
        } else if perform == "classification" {
            if using == "infogain" {
                self.root = id3C(dataset: self.originalDataSet, depth: 0)
            } else if using == "gini" {
                self.root = giniC(dataset: self.originalDataSet, depth: 0)
            } else {
                print("invalid operation requested")
            }
        } else {
            print("invalid operation requested")
        }
        
    }
    
    /// displays the grown tree by calling print tree
    /// - Parameters: None
    /// - Returns: None
    public func displayTree() {
        printTree(node: self.root!, depth: 0)
    }
    
    /// prints part of tree at given node and indents wrt. depth
    /// - Parameters:
    ///   - node: node to be printed
    ///   - depth: depth of the node wrt. root
    /// - Returns: None
    public func printTree(node: Node, depth: Int){
        var indent : String = ""
        for _ in 0 ... depth {
            indent += "  "
        }
        print(indent + node.classification)
        print("\n")
        if (!node.isLeaf){
            for b in node.branches {
                print(indent + b.label)
                printTree(node : b.to, depth: depth+1)
            }
        }
    }
    
    
    /// Forms decision regression tree using gini index recursively
    /// - Parameters:
    ///   - dataset: data left to be used
    ///   - depth: current depth
    /// - Returns: Node that splits data best
    public func giniR(dataset: DataSet, depth: Int) -> Node {
        let currentGiniImpurity = dataset.getGiniImpurity()
        let f = dataset.getGiniFeature()
        
        if currentGiniImpurity != 0.0 &&
           f.giniImpurity! < currentGiniImpurity &&
           depth <= self.maxDepth {
            let node = Node(classification: f.name, isLeaf : false)
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createDataSet(feature : f,
                                             featureValue : value,
                                             data : dataset.data,
                                             target : dataset.target)
                    
                    let gNode = giniR(dataset: data, depth : depth+1)
                    
                    node.addChild(label: value.name, node: gNode)
                }
                return node
            } else {
                let datas = f.getGiniImpurityNumerical(data: dataset.data, target : dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data, column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data, column:  getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = DataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = DataSet(data: d2,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d2))
                
                let leftNode : Node =   giniR(dataset: data1, depth : depth+1)
                let rightNode : Node =   giniR(dataset: data2, depth : depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }
            
        } else {
            let node = Node(classification: String(dataset.getTargetMean()), isLeaf: true)
            return node
        }
        
    }
    
    /// Forms decision classification tree using gini index recursively
    /// - Parameters:
    ///   - dataset: data left to be used
    ///   - depth: current depth
    /// - Returns: Node that splits data best
    public func giniC(dataset: DataSet, depth: Int) -> Node {
        let h = dataset.homogenous()
        
        //if all the classification are the same, creates leaf
        if h.0 {
            let node = Node(classification: h.1, isLeaf: true)
            return node
        }
        
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.data[0].count <= 1 {
            let f = Feature(data: dataset.data, column : 0)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
        let currentGiniImpurity = dataset.getGiniImpurity()
        let f = dataset.getGiniFeature()
        
        if currentGiniImpurity != 0.0 &&
           f.giniImpurity! < currentGiniImpurity &&
           depth < self.maxDepth {
            let node = Node(classification: f.name, isLeaf: false)
            
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createDataSet(feature : f,
                                             featureValue : value,
                                             data : dataset.data,
                                             target : dataset.target)
                    let g_node : Node = giniC(dataset: data, depth : depth+1)
                    node.addChild(label: value.name, node: g_node)
                }
                return node
            } else {
                let datas = f.getGiniImpurityNumerical(data: dataset.data, target : dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column: getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = DataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = DataSet(data: d2,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d2))
                
                let leftNode : Node =   giniC(dataset: data1, depth : depth+1)
                let rightNode : Node =   giniC(dataset: data2, depth : depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }
        } else {
            let f = Feature(data: dataset.data, column : dataset.target)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
    }
    
    
    /// Examine the dataset to create classification Tree with id3 recursively
    /// - Parameters:
    ///   - dataset: data left to be used
    ///   - depth: current depth
    /// - Returns: Node that splits data best
    public func id3C(dataset: DataSet, depth: Int) -> Node{
        
        let h = dataset.homogenous()
        
        //if all the classification are the same, creates leaf
        if h.0 {
            let node = Node(classification: h.1, isLeaf: true)
            return node
        }
        
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.data[0].count == 1 {
            let f = Feature(data: dataset.data, column: 0)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
        //gets best feature to split on and creates a node
        let f = dataset.getMaxInfoGainFeatureForClassification()
        let currentEntropy = dataset.getEntropy()
        
        if currentEntropy != 0.0 &&
           f.entropy! < currentEntropy &&
           depth < self.maxDepth {
            let node = Node(classification: f.name, isLeaf: false)
            
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createDataSet(feature : f,
                                             featureValue : value,
                                             data : dataset.data,
                                             target : dataset.target)
                    
                    let id_node: Node = id3C(dataset: data, depth: depth+1)
                    node.addChild(label: value.name, node: id_node)
                }
                
                return node
            } else {
                let datas = f.getInfoGainNumerical(data: dataset.data, target: dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name,
                                                              data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column:  getColumnNumber(colName: f.name,
                                                               data: datas.2.data))
                
                let data1 = DataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = DataSet(data: d2,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d2))
                
                let leftNode: Node =   id3C(dataset: data1, depth: depth+1)
                let rightNode: Node =   id3C(dataset: data2, depth: depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }
        } else {
            let f = Feature(data: dataset.data, column : dataset.target)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
        //calls id3 on all subset DataSets for all values of the best feature
        
    }
    
    //// Examine the dataset to create regression tree with id3 recursively
    /// - Parameters:
    ///   - dataset: data left to be used
    ///   - depth: current depth
    /// - Returns: Node that splits data best
    public func id3R(dataset: DataSet, depth: Int) -> Node{
        
        //if all the classification are the same, creates leaf
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.getCoeffDev() < self.tolerance || dataset.data[0].count == 1 || dataset.data.count < 4 {
            let node = Node(classification: String(dataset.getTargetMean()), isLeaf: true)
            return node
        }
        
        //gets best feature to split on and creates a node
        let f = dataset.getMaxStdReductionFeature()
        let currentEntropy = dataset.getEntropy()
        
        if currentEntropy != 0.0 && depth <= self.maxDepth {
            
            let node = Node(classification: f.name, isLeaf: false)
            if (Float)(f.values.first!.name) == nil {
                //calls id3 on all subset DataSets for all values of the best feature
                for value in f.values {
                    let data = createDataSet(feature : f,
                                             featureValue : value,
                                             data : dataset.data,
                                             target : dataset.target)
                    let id_node : Node = id3R(dataset: data, depth: depth+1)
                    node.addChild(label: value.name, node: id_node)
                }
                
                return node
            } else {
                let datas = f.getInfoGainNumerical(data: dataset.data, target : dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column:  getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = DataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = DataSet(data: d2,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d2))
                
                let leftNode: Node =   id3R(dataset: data1, depth : depth+1)
                let rightNode: Node =   id3R(dataset: data2, depth : depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }
            
        } else {
            
            let node = Node(classification: String(dataset.getTargetMean()), isLeaf: true)
            return node
            
        }
        
        
    }
    
    /// Classfies/Predicts an example by traversing
    /// - Parameters:
    ///   -example: String array with feature header to be classified
    /// - Returns: classification/predictions as a string
    public func classify(example: [[String]]) -> String {
        var currentNode : Node = self.root!
        
        //loop continues till leaf is found
        while !currentNode.isLeaf {
            
            let featureName = currentNode.classification
            let featureCol = getColumnNumber(colName: featureName, data: example)
            let value = example[1][featureCol]
            var newNode = false
            
            if (Float)(value) == nil {
                for branch in currentNode.branches {
                    if branch.label == value {
                        currentNode = branch.to
                        newNode = true
                    }
                }
            } else {
                //print(currentNode.cutoff, value)
                if (Float)(value)! <= currentNode.cutoff! {
                    currentNode = currentNode.branches.last!.to
                    newNode = true
                } else {
                    currentNode = currentNode.branches.first!.to
                    newNode = true
                }
                
                if !newNode {
                    return "unknown categorical variable"
                }
            }
        }
        
        return currentNode.classification
        
    }
    
    /// Scores the tree's accuracy on test data
    ///
    /// - Parameters:
    ///   - testData: test data as a 2D string array with feature header
    /// - Returns:
    ///   - accuracy of classifications as float
    ///   - classifications as string array
    public func score(testData: [[String]]) -> (Float, [String]){
        
        let head = testData[0]
        
        if self.perform == "regression" {
            
            var residualsSquared: Float = 0
            var predictions = [String]()

            for i in 1 ... testData.endIndex-1 {
                var currentSample = [[String]]()
                currentSample.append(head)
                currentSample.append(testData[i])
                
                let prediction = classify(example: currentSample)
                predictions.append(prediction)
                let residual = (Float)(prediction)! - (Float)(testData[i][target])!
                
                residualsSquared = residualsSquared + (residual*residual)
            }
            
            let RMSE = pow(residualsSquared/(Float)(testData.endIndex-1), 0.5)
            
            return (RMSE, predictions)
        } else {
            var correctClassification = 0
            var classifications = [String]()

            for i in 1 ... testData.endIndex-1 {
                var currentSample = [[String]]()
                currentSample.append(head)
                currentSample.append(testData[i])
                
                let classification = classify(example: currentSample)
                classifications.append(classification)

                if classification == testData[i][target] {
                    correctClassification = correctClassification + 1
                }
            }
            return ((Float)(correctClassification)/(Float)(testData.endIndex-1), classifications)
        }
    }
    
}

/// Deletes a column in a given 2D array | Helper function
/// - Parameters:
///   - data: array from which column is to be deleted
///   - column: col num to be deleted
/// - Returns: array with column deleted
public func deleteColumn(data: [[String]], column: Int) -> [[String]] {
    var modified : [[String]]
    modified = []
    
    for i in stride(from: 0, through: data.count-1, by: 1){
        var d : [String]
        d = []
        for j in stride(from: 0, through: data[i].count-1, by: 1){
            if j == column {
                continue
            }
            d.append(data[i][j])
        }
        modified.append(d)
    }
    return modified
}

/// Returns the column number for a given column name | Helper Function
/// - Parameters:
///   - colName: name of the feature at the column
///   - data: array getting the col num from
/// - Returns: column number for the given column name
public func getColumnNumber(colName: String, data: [[String]]) -> Int{
    var col = -1
    for i in stride(from: 0, through: data[0].count-1, by: 1){
        if data[0][i] == colName{
            col = i
            return col
        }
    }
    return col
}

/// Creates a subset DataSet where all examples have a specific feature value | Helper Function
/// - Parameters:
///   - feature: feature we are targeting
///   - featureValue: feature value desired
///   - data: current data
///   - target: col num for target var
/// - Returns: subset DataSet where all examples have the desired feature value
public func createDataSet(feature: Feature,
                          featureValue: FeatureValue,
                          data: [[String]],
                          target: Int) -> DataSet{
    let c = getColumnNumber(colName : feature.name, data: data)
    var mod = [[String]]()
    mod.append(data[0])
    
    for i in stride(from: 1, through: data.count-1, by: 1){
        if data[i][c] == featureValue.name {
            mod.append(data[i])
        }
    }
    let d = deleteColumn(data: mod, column: c)
    let targetName : String = data[0][target]
    let t = getColumnNumber(colName: targetName, data: d)
    let dataSet = DataSet(data: d, target: t)
    
    return dataSet
}


/// Splits a given dataset into two at a index | Helper Function
/// - Parameters:
///   - data: data tp be split
///   - startIndex: split index - value included in second dataset returned
/// - Returns: Two string arrays split from original arrayß
public func splitDataSet(data: [[String]], startIndex: Int) -> ([[String]], [[String]]) {
    let title = data[0]
    var firstPart = Array(data[1..<startIndex])
    var secondPart = Array(data[startIndex..<data.count])
    
    firstPart.insert(title, at:0)
    secondPart.insert(title, at:0)
    return(firstPart,secondPart)
}

//A node in the decision tree
public class Node {
    /// feature name or value (if leaf) stored in the node
    public var classification: String
    /// whether node is a leaf or not
    public var isLeaf: Bool
    /// parent node of the node
    public weak var parent: Node?
    /// used for splitting continous vars
    public var cutoff: Float?
    /// array of branches ot children
    public var branches: [Branch] = [Branch]()
    
    /// creates a node
    /// - Parameters:
    ///   - classification: feature name or value (if leaf) stored in the node
    ///   - isLeaf: whether node is a leaf or not
    /// - Returns: Node
    public init(classification: String, isLeaf: Bool) {
        self.classification = classification
        self.isLeaf = isLeaf
    }
    
    /// Adds a child by linking two nodes with a branch
    /// - Parameters:
    ///   - label: feature value
    ///   - node: the child node
    /// - Returns: None
    public func addChild(label: String, node: Node) {
        let b = Branch(label: label, from: self, to: node)
        node.parent = self
        self.branches.append(b)
    }
    
    /// Adds a fork by linking two nodes with a branch - used for continous vars
    /// - Parameters:
    ///   - label: feature value
    ///   - node: the child node
    ///   - cutoff: more or less than cutoff values
    /// - Returns: None
    public func addFork(label: String, node: Node, cutoff: Float) {
        let b = Branch(label: label, from: self, to: node)
        self.cutoff = cutoff
        node.parent = self
        self.branches.append(b)
    }
}

/// Branch between two nodes
public class Branch {
    
    /// a feature value of the from node
    public var label: String
    /// parent node
    public var from: Node
    /// child node
    public var to: Node
    
    /// Creates a Branch
    /// - Parameters:
    ///   - label: feature value of the from node
    ///   - from: parent node
    ///   - to: child node
    /// - Returns: Branch
    public init(label: String, from: Node, to: Node){
        self.label = label
        self.from = from
        self.to = to
    }
}


public class DataSet {
    
    /// data
    public var data: [[String]]
    /// entropy of the dataset
    public var entropy: Float?
    /// infoGains provided by each feature
    public var infoGains: Dictionary<Feature, Float>
    /// best feature to use to grow tree
    public var splitFeature: Feature
    /// col num of target var
    public var target: Int
    /// standard deviation
    public var stdDev: Float
    /// giniImpurity of the data set
    public var giniImpurity : Float?
    
    
    /// Creates a DataSet
    ///  - Parameters:
    ///    - data: data with labels
    ///    - target: column number of label
    ///  - Returns: DataSet
    public init(data: [[String]], target: Int){
        self.data = data
        self.stdDev = 0.0
        self.target = target
        self.infoGains = Dictionary<Feature, Float>()
        if target != 0 {
            self.splitFeature = Feature(data: self.data, column : 0)
        } else if data[0].count > 1{
            self.splitFeature = Feature(data: self.data, column : 1)
        } else {
            self.splitFeature = Feature(data: self.data, column : 0)
        }
    }
    
    /// Returns if dataset has same target classification for all examples
    /// - Returns:
    ///   - if dataset has same target classification for all examples
    ///   - the target classification
    public func homogenous() -> (Bool, String) {
        let classification : String = self.data[1][self.target]
        for i in stride(from: 1, through: data.count-1, by:1){
            if self.data[i][self.target] != classification {
                return (false, classification)
            }
        }
        return (true, classification)
    }
    
    /// Returns Coefficient of Standard Deviation
    /// - Parameters: None
    /// - Returns: Coefficient of Standard Deviation as Float
    public func getCoeffDev() -> Float {
        return (getTargetStdDev()/getTargetMean())*100
    }
    
    /// Returns standard deviation of continous target variable
    /// - Parameters: None
    /// - Returns: standard deviation of target variable as Float
    public func getTargetStdDev() -> Float{
        let t = Feature(data: self.data, column: self.target)
        var sd: Float = 0.0
        let total = self.data.count - 1
        let mean = getTargetMean()
        var s: Float = 0.0
        for value in t.values {
            let number = (Float)(value.name)!-mean
            s += (pow(number, 2))*(Float)(value.occurences)
        }
        sd = Float((s/(Float)(total)).squareRoot())
        return sd
    }
    
    /// Returns mean of continous target variable
    /// - Parameters: None
    /// - Returns: mean of target variable as Float
    public func getTargetMean() -> Float{
        let t = Feature(data: self.data, column: self.target)
        var mean: Float = 0.0
        let total = self.data.count - 1
        var count: Float = 0.0
        for value in t.values {
            count += (Float)(value.name)!*(Float)(value.occurences)
        }
        mean = count/(Float)(total)
        return mean
    }
    
    /// Returns entropy of dataset and sets entropy
    /// - Parameters: None
    /// - Returns: entropy of dataset as Float
    public func getEntropy() -> Float{
        let t = Feature(data: self.data, column: self.target)
        var e: Float = 0.0
        let total = self.data.count - 1
        for value in t.values {
            let number = (Float)(value.occurences)/(Float)(total)
            e += -1 * number * log2(number)
        }
        self.entropy = e
        return e
    }
    
    /// Returns the bestFeature with max infoGain to be used in id3
    /// - Parameters: None
    /// - Returns: best feature
    public func getMaxInfoGainFeatureForClassification() -> Feature {
        var bestEntropy: Float
        if (Float)(self.splitFeature.values.first!.name) == nil {
            bestEntropy = self.splitFeature.getInfoGainCategorical(data: self.data,
                                                                    target : self.target).0
        } else {
            bestEntropy = self.splitFeature.getInfoGainNumerical(data: self.data,
                                                                  target : self.target).0
        }
        let datasetEntropy = self.getEntropy()
        self.splitFeature.entropy = bestEntropy
        var bestInfoGain: Float =  datasetEntropy - bestEntropy
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f = Feature(data: self.data, column: i)
                var featureEntropy: Float
                
                if (Float)(f.values.first!.name) == nil {
                    featureEntropy = f.getInfoGainCategorical(data: self.data,
                                                            target: self.target).1
                } else {
                    featureEntropy = f.getInfoGainNumerical(data: self.data,
                                                            target: self.target).0
                }
                let infoGain: Float = datasetEntropy - featureEntropy
                if infoGain > bestInfoGain {
                    bestInfoGain = infoGain
                    self.splitFeature = f
                    self.splitFeature.entropy = featureEntropy
                }
            }
        }
        
        return self.splitFeature
    }
    
    /// Returns the best gini feature i.e. minimum gini impurity
    /// - Parameters: None
    /// - Returns: best feature
    public func getGiniFeature() -> Feature {
        
        var bestImpurity: Float
        
        if (Float)(self.splitFeature.values.first!.name) == nil {
            bestImpurity = self.splitFeature.getGiniImpurityCategorical(data: self.data,
                                                                        target: self.target)
        } else {
            bestImpurity = self.splitFeature.getGiniImpurityNumerical(data: self.data,
                                                                      target: self.target).0
        }
        
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f: Feature = Feature(data: self.data, column: i)
                var impurity: Float
                if (Float)(f.values.first!.name) == nil {
                    impurity = f.getGiniImpurityCategorical(data: self.data,
                                                            target: self.target)
                } else {
                    impurity = f.getGiniImpurityNumerical(data: self.data,
                                                          target: self.target).0
                }
                if bestImpurity > impurity {
                    bestImpurity = impurity
                    self.splitFeature = f
                    self.splitFeature.giniImpurity = impurity
                }
              }
          }
          return self.splitFeature
    }
    
    
    /// Returns the feature with most standard deviation reduction
    /// - Parameters: None
    /// - Returns: feature
    public func getMaxStdReductionFeature() -> Feature {
        let bestSD: Float = self.splitFeature.getTargetStdDev(data: self.data,
                                                              target: self.target)
        var bestSDR: Float = getTargetStdDev() - bestSD
        
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f: Feature = Feature(data: self.data, column: i)
                let sdt: Float = f.getTargetStdDev(data: self.data,
                                                    target: self.target)
                let sdr: Float = getTargetStdDev() - sdt
                
                if sdr > bestSDR {
                    bestSDR = sdr
                    self.splitFeature = f
                }
            }
        }
        return self.splitFeature
    }
    
    /// Returns gini impurity of the data set
    /// - Parameters: None
    /// - Returns: gini impurity of the data set as a Float
    public func getGiniImpurity() -> Float {
        let t : Feature = Feature(data: self.data, column: self.target)
        var i : Float = 0.0
        let total = self.data.count - 1
        
        for value in t.values {
            let number = (Float)(value.occurences)/(Float)(total)
            i += pow(number,2)
        }
        
        self.giniImpurity = 1-i
        return 1-i
    }
    
}

public class Feature: Hashable {
    
    /// Feature Name
    public var name: String
    /// Possible values for the feature
    public var values: Set<FeatureValue>
    /// entropy of dataset if feature is chosen to grow tree
    public var entropy: Float?
    /// gini impurity of dataset if feature is chosen to grow tree
    public var giniImpurity: Float?
    
    /// Initializer for Feature
    /// - Parameters:
    ///   - data: 2D string of data
    ///   - column: column number of feature
    /// - Returns: Feature
    public init(data: [[String]], column: Int){
        self.name = data[0][column]
        self.values = Set<FeatureValue>()
        for i in stride(from: 1, through: data.count-1, by: 1) {
            let val = data[i][column]
            let v = FeatureValue(name: val, occurences: 0)
            if(values.contains(v)){
                let removed = self.values.remove(v)!
                let add = FeatureValue(name: val,
                                       occurences: removed.occurences+1)
                self.values.insert(add)
            } else {
                let add = FeatureValue(name: val, occurences: 1)
                self.values.insert(add)
            }
        }
        
    }
    
    /// Returns most occuring featureValue
    /// - Parameters: None
    /// - Returns: FeatureValue
    public func getDominantValue() -> FeatureValue{
        var dominantV = values.first!
        for v in values {
            if v.occurences > dominantV.occurences {
                dominantV = v
            }
        }
        return dominantV
    }
    
    /// Returns standard deviation of continous target variable
    /// - Parameters:
    ///   - data: string array of data
    ///   - target: col num of target var
    /// - Returns: standard deviation of target variable as Float
    public func getTargetStdDev(data: [[String]], target: Int) -> Float{
        var i: [Float] = []
        let total = data.count-1
        for v in self.values {
            let dataset = createDataSet(feature: self,
                                            featureValue: v,
                                            data: data,
                                            target: target)
            let sdt : Float = dataset.getTargetStdDev()
            let number : Float = (Float)(v.occurences)/(Float)(total)
            i.append(number*sdt)
        }
        var sd : Float = 0.0
        for info in i {
            sd += info
        }
        return sd
    }
    
    /// Computes info gain values for categorical feature/feature values
    /// - Parameters:
    ///   - data: string array of data
    ///   - target: col num of target var
    /// - Returns:
    ///   - infoGain: from choosing the feature as float
    ///   - featureEntropy: entropy of the feature as float
    public func getInfoGainCategorical(data: [[String]], target: Int) -> (Float, Float){
        var entropies: [Float] = []
        let total = data.count-1
        for v in self.values {
            let dataset = createDataSet(feature: self,
                                        featureValue: v,
                                        data: data,
                                        target: target)
            let e: Float = dataset.getEntropy()
            let number: Float = (Float)(v.occurences)/(Float)(total)
            entropies.append(number*e)
        }
        var featureEntropy: Float = 0.0
        for entropy in entropies {
            featureEntropy += entropy
        }
        let baseDataSet = DataSet(data: data,target: target)
        let baseEntropy = baseDataSet.getEntropy()
        
        let infoGain = baseEntropy - featureEntropy
        
        return (infoGain, featureEntropy)
    }
    
    /// Computes info gain values for numerical feature/feature values
    /// - Parameters:
    ///   - data: string array of data
    ///   - target: col num of target var
    /// - Returns:
    ///   - entropy: entropy of the feature
    ///   - dataset1: dataset split part 1
    ///   - dataset2: dataset split part 2
    ///   - cutoff: for the dataset split
    public func getInfoGainNumerical(data: [[String]], target: Int) -> (Float, DataSet, DataSet, Float) {
        let title = data[0]
        var mod = Array(data[1...data.count-1])
        let col: Int = getColumnNumber(colName: self.name, data: data)
        
        mod.sort { left, right in
            (Float)(left[col])! < (Float)(right[col])!
        }
        mod.insert(title, at:0)
        
        var split = splitDataSet(data: mod, startIndex: 2)
        var data1 = DataSet(data: split.0, target: target)
        var data2 = DataSet(data: split.1, target: target)
        var entropy = min(data1.getEntropy(), data2.getEntropy())
        
        for i in stride(from: 2, through: data.count-1, by: 1){
            split = splitDataSet(data: mod, startIndex: i)
            let d1 = DataSet(data: split.0, target: target)
            let d2 = DataSet(data: split.1, target: target)
            let newEntropy = min(d1.getEntropy(), d2.getEntropy())
            
            if newEntropy < entropy {
                entropy = newEntropy
                data1 = d1
                data2 = d2
            }
        }
        
        self.entropy = entropy
        return (entropy, data1, data2, (Float)(data1.data.last![getColumnNumber(colName: self.name, data: data1.data)])!)
    }
    
    
    /// Computes gini impurity values for categorical feature/feature values
    /// - Parameters:
    ///   - data: string array of data
    ///   - target: col num of target var
    /// - Returns: gini impurity of choosing feature as float
    public func getGiniImpurityCategorical(data: [[String]], target: Int) -> Float {
        var giniImp : Float = 0.0
        let total = data.count-1
        for value in self.values {
            let dataset = createDataSet(feature: self,
                                            featureValue: value,
                                            data: data,
                                            target: target)
            let i: Float = dataset.getGiniImpurity()
            giniImp += i*((Float)(value.occurences)/(Float)(total))
        }
        self.giniImpurity = giniImp
        return giniImp
    }
    
    /// Computes gini impurity values for numerical feature/feature values
    /// - Parameters:
    ///   - data: string array of data
    ///   - target: col num of target var
    /// - Returns:
    ///   - giniImpurity: gini impurity of the feature
    ///   - dataset1: dataset split part 1
    ///   - dataset2: dataset split part 2
    ///   - cutoff: for the dataset split
    public func getGiniImpurityNumerical(data: [[String]], target: Int) -> (Float, DataSet, DataSet, Float) {
        let title = data[0]
        var mod = Array(data[1...data.count-1])
        let col : Int = getColumnNumber(colName: self.name, data: data)
        
        mod.sort { left, right in
            (Float)(left[col])! < (Float)(right[col])!
        }
        mod.insert(title, at:0)
        
        _ = data.count-1
        var split = splitDataSet(data: mod, startIndex: 2)
        var data1 = DataSet(data: split.0, target: target)
        var data2 = DataSet(data: split.1, target: target)
        var giniI = max(data1.getGiniImpurity(), data2.getGiniImpurity())
        
        
        for i in stride(from: 2, through: data.count-1, by: 1){
            split = splitDataSet(data: mod, startIndex: i)
            
            let d1 = DataSet(data: split.0, target: target)
            let d2 = DataSet(data: split.1, target: target)
            let impurity = max(d1.getGiniImpurity(), d2.getGiniImpurity())
            
            if impurity < giniI {
                giniI = impurity
                data1 = d1
                data2 = d2
            }
        }
        
        self.giniImpurity = giniI
        return (giniI, data1, data2, (Float)(data1.data.last![getColumnNumber(colName: self.name, data: data1.data)])!)
    }
    
    /// Required == func for implementing Hashable
    /// - Parameters:
    ///   - lhs: feature on the left
    ///   - rhs: feature on the right
    /// - Returns: true if feature names is same else false
    public static func ==(lhs: Feature, rhs: Feature) -> Bool{
        if lhs.name != rhs.name {
            return false
        }
        return true
    }
    
    /// Empty required hash func for implementing Hashable
    public func hash(into hasher: inout Hasher) {
    }
    
}

public class FeatureValue: Hashable {
    /// Name of the feature value
    public var name: String
    /// Number of the occurences of feature value
    public var occurences: Int
    
    /// initialzer for Feature Value
    /// - Parameters:
    ///   - name: of the feature value
    ///   - occurences: num of the occurences of feature value
    /// - Returns: FeatureValue
    public init(name: String, occurences: Int){
        self.name = name
        self.occurences = occurences
    }
    
    /// Required == func for implementing Hashable
    /// - Parameters:
    ///   - lhs: featurevalue on the left
    ///   - rhs: featurevalue on the right
    /// - Returns: true if featurevalue names is same else false
    public static func ==(lhs: FeatureValue, rhs: FeatureValue) -> Bool{
        if lhs.name != rhs.name {
            return false
        }
        return true
    }
    
    /// Empty required hash func for implementing Hashable
    public func hash(into hasher: inout Hasher) {
    }
}
