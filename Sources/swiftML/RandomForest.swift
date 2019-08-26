//
//  RandomForest.swift
//  SwiftML
//
//  Created by Victor A on 8/15/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

import Foundation

public class RandomForest{
    
    public var forest: [RandomTree]
    public var fullData: [[String]]
    public var target: Int
    public var nTrees: Int
    public var nFeatures: Int
    public var depth: Int
    public var perform: String
    public var using: String
    public var tolerance: Float
    

    
    /// Initializer for Random Forest
    /// - Parameters:
    ///   - data: data with labels
    ///   - target: label column number
    ///   - perform: regression or classification
    ///   - using: giniIndex or infoGain
    ///   - nTrees: number of Trees
    /// - Returns: Random Forest
    public init (data: [[String]],
                 target: Int, perform: String,
                 using: String, nTrees: Int,
                 nFeatures: Int, depth: Int,
                 tolerance: Float = 0.1) {
        self.fullData = data
        self.target = target
        self.nTrees = nTrees
        self.nFeatures = nFeatures
        self.depth = depth
        self.forest = [RandomTree]()
        self.perform = perform
        self.using = using
        self.tolerance = tolerance
    }
    
    /// Makes the forest using bootstrapped data
    /// - Parameters: None
    /// - Returns: None
    public func make() {
        var bootData: [[String]]
        var outOfBootData: [[String]]
        
        for _ in 0...nTrees {
            let data = splitData(from: self.fullData)
            bootData = data.0
            outOfBootData = data.1
            let tree = RandomTree(data: bootData,
                                  target: self.target,
                                  perform: self.perform,
                                  using: self.using,
                                  with: self.nFeatures,
                                  tolerance: self.tolerance)
            self.forest.append(tree)
        }
    }
    
    /// Returns bootstrapped and out of bootData
    /// - Parameters:
    ///   - from: string array of data
    /// - Returns: boostrapped data and out of boot data
    public func splitData(from: [[String]]) -> ([[String]],[[String]]) {
        var selectedIndices = [Int]()
        let head = from[0]
        var bootstrappedData = [[String]]()
        bootstrappedData.append(head)
        
        for _ in 0...from.count-1 {
            let index = Int.random(in: 1 ... from.count-1)
            bootstrappedData.append(from[index])
            selectedIndices.append(index)
        }
        
        var testData = [[String]]()
        testData.append(head)
        for i in 1...from.count-1 {
            if !selectedIndices.contains(i){
                testData.append(from[i])
            }
        }
        
        return (bootstrappedData, testData)
    }
    
    
    /// Classfies/Predicts an example by using all trees in the foret
    /// - Parameters:
    ///   -this: String array with feature header to be classified
    /// - Returns: classification/predictions as a string
    public func predict(this: [[String]]) -> String{
        if self.perform == "regression"{
            var prediction : Float = 0.0
            for tree in self.forest {
                prediction += (Float)(tree.classify(example:this))!
            }
            return (String)(prediction/(Float)(self.forest.count))
            
        } else {
            let target = Feature(data: self.fullData, column: self.target)
            var classes = Dictionary<String, Int>()
            for value in target.values {
                classes[value.name] = 0
            }
            for tree in self.forest {
                let classification = tree.classify(example:this)
                classes[classification] = classes[classification]! + 1
            }
            let maxClass = classes.max { a, b in a.value < b.value }
            return maxClass!.key
        }
    }
    
    
    /// Scores the random forest's accuracy on test data
    ///
    /// - Parameters:
    ///   - testData: test data as a 2D string array with feature header
    /// - Returns:
    ///   - accuracy of classifications as float
    ///   - classifications as string array
    public func score(with: [[String]]) -> (Float, [String]){
        let head = with[0]
        
        if self.perform == "regression" {
            var residualsSquared : Float = 0
            var predictions = [String]()
            for i in 1 ... with.endIndex-1 {
                var currentSample = [[String]]()
                currentSample.append(head)
                currentSample.append(with[i])
                let prediction = predict(this: currentSample)
                predictions.append(prediction)
                let residual = (Float)(prediction)! - (Float)(with[i][target])!
                residualsSquared = residualsSquared + (residual*residual)
            }
            let RMSE = pow(residualsSquared/(Float)(with.endIndex-1), 0.5)
            return (RMSE, predictions)
        } else {
            var correctClassification = 0
            var classifications = [String]()
            for i in 1 ... with.endIndex-1 {
                var currentSample = [[String]]()
                currentSample.append(head)
                currentSample.append(with[i])
                
                let classification = predict(this: currentSample)
                classifications.append(classification)

                if classification == with[i][target] {
                    correctClassification = correctClassification + 1
                }
            }
            return ((Float)(correctClassification)/(Float)(with.endIndex-1), classifications)
        }
    }
    
}


public class RandomTree {
    
    /// training data set whole as inputed by user in init
    public var originalDataSet: RandomDataSet
    /// root node of the decision tree
    public var root: Node?
    /// max depth tree is grown
    public var maxDepth: Int
    /// number of vars to be used at every step
    public var randomVars: Int
    /// column number of target var
    public var target: Int
    /// tolerance for regression
    public var tolerance: Float
    
    /// Creates original DataSet to be stored and grows decision tree
    ///  - Parameters:
    ///    - data: data with labels
    ///    - target: column number of label
    ///    - perform: regression or classification
    ///    - using: infoGain or giniIndex
    ///    - with: num of random vars to consider at each iteration
    ///  - Returns: DecisionTree
    public init (data: [[String]],
                 target: Int,
                 maxDepth: Int = 9999,
                 perform: String,
                 using: String,
                 with: Int,
                 tolerance: Float = 0.1) {
        self.originalDataSet = RandomDataSet(data: data,
                                             target: target)
        self.maxDepth = maxDepth
        self.randomVars = with
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
    
    /// Returns a set of randomIndices
    /// - Parameters:
    ///   - from: the dataset from which the random index need to be selected
    /// - Returns: Array of random indexes
    public func getRandomIndices(from: RandomDataSet) -> [Int]{
        var randomIndices = [Int]()
        
        for _ in 1 ... min(self.randomVars, from.data[0].count-1) {
            var number = Int.random(in: 0 ..< from.data[0].count)
            
            while number == from.target || randomIndices.contains(number) {
                number = Int.random(in: 0 ..< from.data[0].count)
            }
            
            randomIndices.append(number)
        }
        return randomIndices
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
    public func giniR(dataset: RandomDataSet, depth: Int) -> Node {
        let currentGiniImpurity = dataset.getGiniImpurity()
        let f = dataset.getGiniFeature(fromIndices:getRandomIndices(from: dataset))
        
        if currentGiniImpurity != 0.0 &&
            f.giniImpurity! < currentGiniImpurity &&
            depth <= self.maxDepth {
            let node = Node(classification: f.name, isLeaf : false)
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createRandomDataSet(randomFeature: f,
                                             featureValue: value,
                                             data: dataset.data,
                                             target: dataset.target)
                    
                    let gNode = giniR(dataset: data, depth : depth+1)
                    
                    node.addChild(label: value.name, node: gNode)
                }
                return node
            } else {
                let datas = f.getGiniImpurityNumerical(data: dataset.data, target : dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column: getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = RandomDataSet(data: d1,
                                          target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                                  data: d1))
                let data2 = RandomDataSet(data: d2,
                                          target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                                  data: d2))
                
                let leftNode = giniR(dataset: data1, depth: depth+1)
                let rightNode = giniR(dataset: data2, depth: depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3),
                             node: rightNode,
                             cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3),
                             node: leftNode,
                             cutoff: datas.3)
                
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
    public func giniC(dataset: RandomDataSet, depth: Int) -> Node {
        let h = dataset.homogenous()
        
        //if all the classification are the same, creates leaf
        if h.0 {
            let node = Node(classification: h.1, isLeaf: true)
            return node
        }
        
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.data[0].count <= 1 {
            let f = RandomFeature(data: dataset.data, column : 0)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
        let currentGiniImpurity = dataset.getGiniImpurity()
        let f = dataset.getGiniFeature(fromIndices:getRandomIndices(from: dataset))
        
        if currentGiniImpurity != 0.0 &&
            f.giniImpurity! < currentGiniImpurity &&
            depth < self.maxDepth {
            let node = Node(classification: f.name, isLeaf: false)
            
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createRandomDataSet(randomFeature: f,
                                             featureValue: value,
                                             data: dataset.data,
                                             target: dataset.target)
                    let g_node = giniC(dataset: data, depth : depth+1)
                    node.addChild(label: value.name, node: g_node)
                }
                return node
            } else {
                let datas = f.getGiniImpurityNumerical(data: dataset.data,
                                                       target: dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name,
                                                              data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column: getColumnNumber(colName: f.name,
                                                              data: datas.2.data))
                
                let data1 = RandomDataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = RandomDataSet(data: d2,
                                          target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                                  data: d2))
                
                let leftNode = giniC(dataset: data1, depth : depth+1)
                let rightNode = giniC(dataset: data2, depth : depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3),
                             node: rightNode,
                             cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3),
                             node: leftNode,
                             cutoff: datas.3)
                return node
            }
        } else {
            let f = RandomFeature(data: dataset.data, column : dataset.target)
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
    public func id3C(dataset: RandomDataSet, depth: Int) -> Node{
        
        let h = dataset.homogenous()
        
        //if all the classification are the same, creates leaf
        if h.0 {
            let node = Node(classification: h.1, isLeaf: true)
            return node
        }
        
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.data[0].count == 1 {
            let f = RandomFeature(data: dataset.data, column: 0)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
        //gets best RandomFeature to split on and creates a node
        let f = dataset.getBestFeature(fromIndices:getRandomIndices(from: dataset))
        let currentEntropy = dataset.getEntropy()
        
        if currentEntropy != 0.0 &&
            f.entropy! < currentEntropy &&
            depth < self.maxDepth {
            let node = Node(classification: f.name, isLeaf: false)
            
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createRandomDataSet(randomFeature: f,
                                             featureValue: value,
                                             data: dataset.data,
                                             target: dataset.target)
                    
                    let id_node = id3C(dataset: data, depth: depth+1)
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
                
                let data1 = RandomDataSet(data: d1,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d1))
                let data2 = RandomDataSet(data: d2,
                                    target: getColumnNumber(colName: dataset.data[0][dataset.target],
                                                            data: d2))
                
                let leftNode = id3C(dataset: data1, depth: depth+1)
                let rightNode =   id3C(dataset: data2, depth: depth+1)
                
                node.addFork(label: "more than equal "+(String)(datas.3),
                             node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3),
                             node: leftNode, cutoff: datas.3)
                
                return node
            }
        } else {
            let f = RandomFeature(data: dataset.data, column: dataset.target)
            let v = f.getDominantValue()
            let node = Node(classification: v.name, isLeaf: true)
            return node
        }
        
    }
    
    //// Examine the dataset to create regression tree with id3 recursively
    /// - Parameters:
    ///   - dataset: data left to be used
    ///   - depth: current depth
    /// - Returns: Node that splits data best
    public func id3R(dataset: RandomDataSet, depth: Int) -> Node{
        
        //if all the classification are the same, creates leaf
        //if no non-target attributes are left, creates leaf with dominant class
        if dataset.getCoeffDev() < self.tolerance || dataset.data[0].count == 1 || dataset.data.count < 4 {
            let node = Node(classification: String(dataset.getTargetMean()), isLeaf: true)
            return node
        }
        
        //gets best RandomFeature to split on and creates a node
        let f = dataset.getSplitFeature(fromIndices:getRandomIndices(from: dataset))
        let currentEntropy = dataset.getEntropy()
        
        if currentEntropy != 0.0 && depth <= self.maxDepth {
            
            let node = Node(classification: f.name, isLeaf: false)
            if (Float)(f.values.first!.name) == nil {
                //calls id3 on all subset DataSets for all values of the best feature
                for value in f.values {
                    let data = createRandomDataSet(randomFeature: f,
                                             featureValue: value,
                                             data: dataset.data,
                                             target: dataset.target)
                    let id_node = id3R(dataset: data, depth: depth+1)
                    node.addChild(label: value.name, node: id_node)
                }
                
                return node
            } else {
                let datas = f.getInfoGainNumerical(data: dataset.data, target : dataset.target)
                
                let d1 = deleteColumn(data: datas.1.data,
                                      column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data,
                                      column:  getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = RandomDataSet(data: d1,
                                          target: getColumnNumber(colName: dataset.data[0][dataset.target], data: d1))
                let data2 = RandomDataSet(data: d2,
                                          target: getColumnNumber(colName: dataset.data[0][dataset.target], data: d2))
                
                let leftNode =  id3R(dataset: data1, depth : depth+1)
                let rightNode = id3R(dataset: data2, depth : depth+1)
                
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
        var currentNode: Node = self.root!
        
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
}




/// Creates a subset Random DataSet where all examples have a specific feature value | Helper Function
/// - Parameters:
///   - feature: feature we are targeting
///   - featureValue: feature value desired
///   - data: current data
///   - target: col num for target var
/// - Returns: subset Random DataSet where all examples have the desired feature value
public func createRandomDataSet(randomFeature: RandomFeature,
                                featureValue: FeatureValue,
                                data: [[String]],
                                target: Int) -> RandomDataSet{
    
    let c = getColumnNumber(colName: randomFeature.name, data: data)
    var mod = [[String]]()
    mod.append(data[0])
    
    for i in stride(from: 1, through: data.count-1, by: 1){
        if data[i][c] == featureValue.name {
            mod.append(data[i])
        }
    }
    
    let d: [[String]] = deleteColumn(data: mod, column: c)
    let targetName: String = data[0][target]
    let t: Int = getColumnNumber(colName: targetName, data: d)
    let DataSet = RandomDataSet(data: d, target: t)
    return DataSet
}






public class RandomDataSet {
    
    /// data
    public var data: [[String]]
    /// entropy of the dataset
    public var entropy: Float?
    /// infoGains provided by each feature
    public var infoGains: Dictionary<RandomFeature, Float>
    /// best feature to use to grow tree
    public var splitFeature: RandomFeature
    /// col num of target var
    public var target: Int
    /// standard deviation
    public var stdDev: Float
    /// giniImpurity of the data set
    public var giniImpurity : Float?
    
    /// Creates a Random DataSet
    ///  - Parameters:
    ///    - data: data with labels
    ///    - target: column number of label
    ///  - Returns: DataSet
    public init(data: [[String]], target: Int){
        self.data = data
        self.stdDev = 0.0
        self.target = target
        self.infoGains = Dictionary<RandomFeature, Float>()
        if target != 0 {
            self.splitFeature = RandomFeature(data: self.data, column : 0)
        } else if data[0].count > 1{
            self.splitFeature = RandomFeature(data: self.data, column : 1)
        } else {
            self.splitFeature = RandomFeature(data: self.data, column : 0)
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
        let t : RandomFeature = RandomFeature(data: self.data, column: self.target)
        var sd : Float = 0.0
        let total = self.data.count - 1
        let mean = getTargetMean()
        var s : Float = 0.0
        
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
        let t : RandomFeature = RandomFeature(data: self.data, column: self.target)
        var mean : Float = 0.0
        let total = self.data.count - 1
        var count : Float = 0.0
        
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
        
        let t : RandomFeature = RandomFeature(data: self.data, column: self.target)
        var e : Float = 0.0
        let total = self.data.count - 1
        
        for value in t.values {
            let number = (Float)(value.occurences)/(Float)(total)
            e += -1 * number * log2(number)
        }
        
        self.entropy = e
        return e
    }
    
    /// Returns the bestFeature with max infoGain to be used in id3
    /// - Parameters:
    ///   - fromIndices: set of indices of features to be considered
    /// - Returns: best feature
    public func getBestFeature(fromIndices: [Int]) -> RandomFeature {
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
        for i in fromIndices{
            let f = RandomFeature(data: self.data, column: i)
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
        
        return self.splitFeature
    }
    
    /// Returns the best gini feature i.e. minimum gini impurity
    /// - Parameters:
    ///   - fromIndices: set of indices of features to be considered
    /// - Returns: best feature
    public func getGiniFeature(fromIndices: [Int]) -> RandomFeature {
        var bestImpurity: Float
        var bestFeature = RandomFeature(data: self.data, column: fromIndices.first!)
        
        if (Float)(bestFeature.values.first!.name) == nil {
            bestImpurity = bestFeature.getGiniImpurityCategorical(data: self.data,
                                                               target : self.target)
        } else {
            bestImpurity = bestFeature.getGiniImpurityNumerical(data: self.data,
                                                             target : self.target).0
        }
        
        for i in fromIndices{
            let f = RandomFeature(data: self.data, column: i)
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
                bestFeature = f
                bestFeature.giniImpurity = impurity
            }
        }
        self.splitFeature = bestFeature
        return bestFeature
    }
    
    /// Returns the feature with most standard deviation reduction
    /// - Parameters:
    ///   - fromIndices: set of indices of features to be considered
    /// - Returns: feature
    public func getSplitFeature(fromIndices: [Int]) -> RandomFeature {
        var bestFeature = RandomFeature(data: self.data, column: fromIndices.first!)
        let bestSD : Float = bestFeature.getTargetStdDev(data: self.data,
                                                         target: self.target)
        var bestSDR : Float = getTargetStdDev() - bestSD
        
        for i in fromIndices{
            let f: RandomFeature = RandomFeature(data: self.data, column: i)
            let sdt: Float = f.getTargetStdDev(data: self.data,
                                                target: self.target)
            let sdr: Float = getTargetStdDev() - sdt
            
            if sdr > bestSDR {
                bestSDR = sdr
                bestFeature = f
            }
        }
        
        return bestFeature
    }
    
    
    /// Returns gini impurity of the data set
    /// - Parameters: None
    /// - Returns: gini impurity of the data set as a Float
    public func getGiniImpurity() -> Float {
        let t : RandomFeature = RandomFeature(data: self.data, column: self.target)
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

public class RandomFeature: Hashable {
    
    public var name: String
    public var values: Set<FeatureValue>
    public var entropy: Float?
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
            let dataset = createRandomDataSet(randomFeature: self,
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
            let dataset = createRandomDataSet(randomFeature: self,
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
    public func getInfoGainNumerical(data: [[String]], target: Int) -> (Float, RandomDataSet, RandomDataSet, Float) {
        let title = data[0]
        var mod = Array(data[1...data.count-1])
        let col: Int = getColumnNumber(colName: self.name, data: data)
        
        mod.sort { left, right in
            (Float)(left[col])! < (Float)(right[col])!
        }
        mod.insert(title, at:0)
        
        var split = splitDataSet(data: mod, startIndex: 2)
        var data1 = RandomDataSet(data: split.0, target: target)
        var data2 = RandomDataSet(data: split.1, target: target)
        var entropy = min(data1.getEntropy(), data2.getEntropy())
        
        for i in stride(from: 2, through: data.count-1, by: 1){
            split = splitDataSet(data: mod, startIndex: i)
            let d1 = RandomDataSet(data: split.0, target: target)
            let d2 = RandomDataSet(data: split.1, target: target)
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
            let dataset = createRandomDataSet(randomFeature: self,
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
    public func getGiniImpurityNumerical(data: [[String]], target: Int) -> (Float, RandomDataSet, RandomDataSet, Float) {
        let title = data[0]
        var mod = Array(data[1...data.count-1])
        let col : Int = getColumnNumber(colName: self.name, data: data)
        
        mod.sort { left, right in
            (Float)(left[col])! < (Float)(right[col])!
        }
        mod.insert(title, at:0)
        
        var split = splitDataSet(data: mod, startIndex: 2)
        var data1 = RandomDataSet(data: split.0, target: target)
        var data2 = RandomDataSet(data: split.1, target: target)
        var giniI = max(data1.getGiniImpurity(), data2.getGiniImpurity())
        
        
        for i in stride(from: 2, through: data.count-1, by: 1){
            split = splitDataSet(data: mod, startIndex: i)
            
            let d1 = RandomDataSet(data: split.0, target: target)
            let d2 = RandomDataSet(data: split.1, target: target)
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
    public static func ==(lhs: RandomFeature, rhs: RandomFeature) -> Bool{
        if lhs.name != rhs.name {
            return false
        }
        return true
    }
    
    /// Empty required hash func for implementing Hashable
    public func hash(into hasher: inout Hasher) {
    }
    
}
